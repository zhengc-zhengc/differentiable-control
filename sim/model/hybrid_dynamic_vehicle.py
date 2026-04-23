# sim/model/hybrid_dynamic_vehicle.py
"""混合动力学车辆模型：机理模型 + MLP 残差修正。

将 plant 仓库中训练好的混合架构集成到 sim 仿真链路：
- Base: VehicleDynamics (Euler 积分) — 必须与 MLP 训练时一致
- MLP: 预测 3D 运动残差 [dvx, dvy, dr]，重建为 6D 状态修正
- 前向/反向均可微，支持 BPTT 控制器调参

接口与 BicycleModel / DynamicVehicle 一致：
step(delta, acc), x, y, yaw, v, detach_state()

注意：
- Euler 积分是必须的，MLP 学习的是 Euler base 的残差，RK4 会导致不匹配
- 车辆参数必须与 MLP 训练时一致（corner_stiff=56000, air_density=1.206）
- MLP 权重冻结（不参与控制器优化），但计算图梯度仍流过 MLP 到控制器参数
"""
import math
import os

import torch
import torch.nn as nn

from model.dynamic_vehicle import VehicleDynamics

# MLP 输入中选取的控制量索引：[delta_sw, T_rl, T_rr] → control[:, [0, 3, 4]]
_MLP_CONTROL_INDICES = [0, 3, 4]


class MLPErrorModel(nn.Module):
    """MLP 残差网络。

    结构：Linear(in, 128) -> ReLU -> Dropout(0.1) x3 -> Linear(128, out)
    输出层零初始化（训练从"完全信任 Base 模型"开始）。
    """
    def __init__(self, input_dim: int = 10, output_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _reconstruct_full_error(motion_error, base_next, dt):
    """3D 运动残差 [dvx, dvy, dr] -> 6D 状态修正 [dx, dy, dyaw, dvx, dvy, dr]。

    步骤（与 plant/train.py derive_full_error_from_motion_error_torch 一致）：
    1. vx/vy 误差旋转到世界坐标系（用 base_next 的 yaw）
    2. 位置误差 = 世界速度误差 * dt
    3. 航向误差 = r 误差 * dt
    """
    safe_dt = max(dt, 1.0e-6)
    yaw_ref = base_next[:, 2:3]

    vx_err = motion_error[:, 0:1]
    vy_err = motion_error[:, 1:2]
    r_err = motion_error[:, 2:3]

    cos_yaw = torch.cos(yaw_ref)
    sin_yaw = torch.sin(yaw_ref)

    # 体坐标系速度误差 -> 世界坐标系
    wx_err = cos_yaw * vx_err - sin_yaw * vy_err
    wy_err = sin_yaw * vx_err + cos_yaw * vy_err

    # 积分：速度误差 * dt -> 位置/航向修正
    x_err = wx_err * safe_dt
    y_err = wy_err * safe_dt
    yaw_err = r_err * safe_dt

    pose_err = torch.cat([x_err, y_err, yaw_err], dim=1)
    return torch.cat([pose_err, motion_error], dim=1)


class HybridDynamicVehicle:
    """混合动力学车辆：Base(Euler) + MLP 残差修正。

    内部维护 6D 状态 [x_f, y_f, yaw, vx, vy, r]（前轴参考点），
    对外暴露后轴坐标 x, y（通过 x_r = x_f - L*cos(yaw), y_r = y_f - L*sin(yaw) 转换）。

    每步执行：
    1. Base Euler 积分 -> base_next
    2. 构建 MLP 特征 [state(6), delta_sw(1), T_rl(1), T_rr(1), dt(1)] = 10D
    3. 特征归一化 -> MLP 前向 -> 3D 运动残差
    4. 重建 6D 状态修正
    5. final = base_next + correction
    """
    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0,
                 dt=0.02, differentiable=False, checkpoint_path=None):
        self.params = params
        self.dt = dt
        self.differentiable = differentiable
        self.dynamics = VehicleDynamics(params)
        self._steer_ratio = self.dynamics.steer_ratio

        # 输入 (x, y) 是后轴坐标，转换为前轴坐标存储
        L = self.dynamics.lf + self.dynamics.lr
        yaw_f = float(yaw)
        x_f = float(x) + L * math.cos(yaw_f)
        y_f = float(y) + L * math.sin(yaw_f)

        # 6D 状态：[x_f, y_f, yaw, vx, vy, r]
        self._state = torch.tensor(
            [x_f, y_f, yaw_f, float(v), 0.0, 0.0])

        self._mlp = None
        self._feature_mean = None
        self._feature_scale = None
        self._motion_error_clip = None

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path):
        """加载 MLP checkpoint 及归一化统计量。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MLP checkpoint 不存在: {path}")

        payload = torch.load(path, map_location='cpu', weights_only=False)

        if isinstance(payload, dict) and 'state_dict' in payload:
            state_dict = payload['state_dict']
            input_dim = int(payload.get('model_input_dim', 10))
            output_dim = int(payload.get('model_output_dim', 3))

            if 'feature_mean' in payload and 'feature_scale' in payload:
                self._feature_mean = torch.tensor(
                    payload['feature_mean'], dtype=torch.float32).reshape(1, -1)
                self._feature_scale = torch.tensor(
                    payload['feature_scale'], dtype=torch.float32).reshape(1, -1)

            if 'motion_error_scale' in payload:
                self._motion_error_clip = 3.0 * torch.tensor(
                    payload['motion_error_scale'],
                    dtype=torch.float32).reshape(1, -1)
        else:
            state_dict = payload if isinstance(payload, dict) else {}
            weights = [v for k, v in state_dict.items()
                       if k.endswith('.weight') and v.ndim == 2]
            if not weights:
                raise ValueError("Checkpoint 不包含线性层权重")
            input_dim = int(weights[0].shape[1])
            output_dim = int(weights[-1].shape[0])

        self._mlp = MLPErrorModel(input_dim=input_dim, output_dim=output_dim)
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()

        # 冻结 MLP 权重：不参与控制器调参优化
        for p in self._mlp.parameters():
            p.requires_grad_(False)

    def step(self, delta, torque_wheel):
        """前进一步。

        Args:
            delta: 前轮转角 (rad)
            torque_wheel: 车轮总扭矩 (N·m)，后驱两轮平分
        """
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(float(delta))
        if not isinstance(torque_wheel, torch.Tensor):
            torque_wheel = torch.tensor(float(torque_wheel))

        # 控制量转换：前轮角 -> 方向盘角，后驱两轮平分总扭矩
        delta_sw = delta * self._steer_ratio
        torque_rear = torque_wheel / 2.0
        zero = torch.zeros_like(torque_wheel)

        control = torch.stack(
            [delta_sw, zero, zero, torque_rear, torque_rear]).unsqueeze(0)
        state = self._state.unsqueeze(0)

        # Base: Euler 积分（不能用 RK4，MLP 训练时用的是 Euler）
        derivatives = self.dynamics.derivatives(state, control)
        base_next = state + derivatives * self.dt

        if self._mlp is not None:
            # MLP 特征：[state(6), delta_sw(1), T_rl(1), T_rr(1), dt(1)]
            selected_ctrl = control[:, _MLP_CONTROL_INDICES]
            dt_t = state.new_tensor([[self.dt]])
            features = torch.cat([state, selected_ctrl, dt_t], dim=1)

            if self._feature_mean is not None:
                features = (features - self._feature_mean) / self._feature_scale

            motion_error = self._mlp(features)

            if self._motion_error_clip is not None:
                motion_error = torch.clamp(
                    motion_error,
                    -self._motion_error_clip,
                    self._motion_error_clip)

            full_error = _reconstruct_full_error(
                motion_error, base_next, self.dt)
            self._state = (base_next + full_error).squeeze(0)
        else:
            self._state = base_next.squeeze(0)

    def detach_state(self):
        """截断梯度链（Truncated BPTT）。"""
        self._state = self._state.detach().requires_grad_(False)

    @property
    def x(self):
        """后轴 x 坐标（从前轴内部状态转换）。"""
        x_f = self._state[0]
        yaw = self._state[2]
        L = self.dynamics.lf + self.dynamics.lr
        return x_f - L * torch.cos(yaw)

    @property
    def y(self):
        """后轴 y 坐标（从前轴内部状态转换）。"""
        y_f = self._state[1]
        yaw = self._state[2]
        L = self.dynamics.lf + self.dynamics.lr
        return y_f - L * torch.sin(yaw)

    @property
    def yaw(self):
        return self._state[2]

    @property
    def v(self):
        """合速度 = sqrt(vx^2 + vy^2)。"""
        vx = self._state[3]
        vy = self._state[4]
        return torch.sqrt(vx * vx + vy * vy + 1e-10)

    @property
    def yawrate(self):
        """当前横摆角速度 r (rad/s)，base Euler 积分 + MLP 残差修正后的值。"""
        return self._state[5]

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / math.pi)
