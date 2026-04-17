# sim/model/truck_trailer_vehicle.py
"""牵引车-挂车双体动力学车辆：base RK4 + 可选 MLP 残差修正。

底层动力学 + MLP 网络结构本地化在 `truck_trailer_dynamics.py`（来自上游
truckdynamicmodel 的拷贝），本仓库自包含、可独立运行。

接口与 BicycleModel / DynamicVehicle / HybridDynamicVehicle 一致：
- step(delta, torque_wheel)
- x, y, yaw, v, speed_kph, yaw_deg
- detach_state()

参考点约定：
- 内部 12D 状态：[x_t,y_t,psi_t,vx_t,vy_t,r_t, x_s,y_s,psi_s,vx_s,vy_s,r_s]
  其中 (x_t, y_t) 是**牵引车质心**
- 对外暴露的 x/y/yaw/v：**牵引车后轴**（与 sim 控制器约定一致）
  转换：x_rear = x_t - b_t × cos(yaw)，b_t = L_t - a_t（质心到后轴距离）
"""
import math
import os

import torch

from model.truck_trailer_dynamics import (
    MLPErrorModel,
    TruckTrailerNominalDynamics,
    wrap_angle_error_torch,
)

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# data_utils 中纯 torch 的 2 个特征构造函数，内联避免拖入 pandas/sklearn 依赖

def build_mlp_input_feature_tensor(state, control, trailer_mass_kg, dt):
    """构建 MLP 输入特征：[trailer_mass, vx_t, vy_t, r_t, speed_t,
    vx_s, vy_s, r_s, speed_s, articulation, sin/cos articulation,
    5×control, dt] = 18D"""
    if trailer_mass_kg.ndim == 1:
        trailer_mass_kg = trailer_mass_kg.unsqueeze(1)
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)

    articulation = wrap_angle_error_torch(state[:, 8:9] - state[:, 2:3])
    speed_t = torch.sqrt(
        state[:, 3:4] * state[:, 3:4] + state[:, 4:5] * state[:, 4:5] + 1.0e-8)
    speed_s = torch.sqrt(
        state[:, 9:10] * state[:, 9:10] + state[:, 10:11] * state[:, 10:11]
        + 1.0e-8)
    return torch.cat([
        trailer_mass_kg, state[:, 3:4], state[:, 4:5], state[:, 5:6],
        speed_t, state[:, 9:10], state[:, 10:11], state[:, 11:12], speed_s,
        articulation, torch.sin(articulation), torch.cos(articulation),
        control, dt,
    ], dim=1)


def derive_full_error_from_motion_error_torch(motion_error, base_next, dt):
    """6D 运动残差 → 12D 状态修正。"""
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    safe_dt = torch.clamp(dt, min=1.0e-6)
    yaw_t = base_next[:, 2:3]
    yaw_s = base_next[:, 8:9]
    evx_t, evy_t, er_t = motion_error[:, 0:1], motion_error[:, 1:2], motion_error[:, 2:3]
    evx_s, evy_s, er_s = motion_error[:, 3:4], motion_error[:, 4:5], motion_error[:, 5:6]

    dx_t = (torch.cos(yaw_t) * evx_t - torch.sin(yaw_t) * evy_t) * safe_dt
    dy_t = (torch.sin(yaw_t) * evx_t + torch.cos(yaw_t) * evy_t) * safe_dt
    dpsi_t = wrap_angle_error_torch(er_t * safe_dt)

    dx_s = (torch.cos(yaw_s) * evx_s - torch.sin(yaw_s) * evy_s) * safe_dt
    dy_s = (torch.sin(yaw_s) * evx_s + torch.cos(yaw_s) * evy_s) * safe_dt
    dpsi_s = wrap_angle_error_torch(er_s * safe_dt)

    return torch.cat(
        [dx_t, dy_t, dpsi_t, evx_t, evy_t, er_t,
         dx_s, dy_s, dpsi_s, evx_s, evy_s, er_s], dim=1)


def _resolve_checkpoint_path(rel_or_abs):
    """checkpoint 路径解析：绝对路径直接用，相对路径相对 sim/ 目录。"""
    if not rel_or_abs:
        return None
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(_SIM_DIR, rel_or_abs))


class TruckTrailerVehicle:
    """牵引车-挂车双体动力学适配器。

    内部维护 12D 状态（牵引车 6 + 挂车 6），用 TruckTrailerNominalDynamics
    做 RK4 积分。可选加载 MLP checkpoint 做 6D 运动残差修正。

    每步执行：
    1. Base RK4 积分 → base_next (12D)
    2. 若有 MLP：构建特征 [trailer_mass, 8 状态量, 3 articulation, 5 control, dt] = 18D
       → MLP 输出 6D 运动残差 → 重建为 12D 状态修正 → final = base_next + correction
    3. 包角到 [-π, π]
    """

    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0,
                 dt=0.02, differentiable=False,
                 checkpoint_path=None, trailer_mass_kg=None):
        self.params = params
        self.dt = dt
        self.differentiable = differentiable
        self.dynamics = TruckTrailerNominalDynamics(params)

        self._steer_ratio = float(params['steering_ratio'])
        self._L_t = float(params['L_t'])
        self._a_t = float(params['a_t'])
        # b_t = 质心到后轴距离
        self._b_t = self._L_t - self._a_t

        # 挂车质量（运行时可变，默认从 params 读）
        if trailer_mass_kg is None:
            trailer_mass_kg = float(params.get(
                'default_trailer_mass_kg', params['m_s_base']))
        self._trailer_mass = float(trailer_mass_kg)

        # 输入 (x, y) 是后轴坐标，转换为质心坐标存储
        yaw_f = float(yaw)
        v_f = float(v)
        x_t_cg = float(x) + self._b_t * math.cos(yaw_f)
        y_t_cg = float(y) + self._b_t * math.sin(yaw_f)

        # 初始挂车位置：稳态铰接（同朝向，铰接点对齐）
        # 铰接点全局 = 牵引车质心 + (hitch_x, hitch_y) 旋转到全局
        # 挂车质心 = 铰接点 - c_s × (挂车朝向)（来自 base_model.py L113-114）
        c_s = float(params['c_s'])
        hitch_x_off = float(params['hitch_x'])
        hitch_y_off = float(params['hitch_y'])
        x_hitch = (x_t_cg + hitch_x_off * math.cos(yaw_f)
                   - hitch_y_off * math.sin(yaw_f))
        y_hitch = (y_t_cg + hitch_x_off * math.sin(yaw_f)
                   + hitch_y_off * math.cos(yaw_f))
        x_s_cg = x_hitch - c_s * math.cos(yaw_f)
        y_s_cg = y_hitch - c_s * math.sin(yaw_f)

        # 12D 状态：牵引车质心 + 挂车质心，初始稳态（vy=0, r=0）
        self._state = torch.tensor([
            x_t_cg, y_t_cg, yaw_f, v_f, 0.0, 0.0,
            x_s_cg, y_s_cg, yaw_f, v_f, 0.0, 0.0,
        ], dtype=torch.float32)

        self._mlp = None
        self._feature_mean = None
        self._feature_scale = None
        self._motion_error_clip = None

        ckpt = _resolve_checkpoint_path(checkpoint_path)
        if ckpt:
            self._load_checkpoint(ckpt)

    def _load_checkpoint(self, path):
        """加载 MLP checkpoint 及归一化统计量。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MLP checkpoint 不存在: {path}")

        payload = torch.load(path, map_location='cpu', weights_only=False)

        if isinstance(payload, dict) and 'state_dict' in payload:
            state_dict = payload['state_dict']
            input_dim = int(payload.get('model_input_dim', 18))
            output_dim = int(payload.get('model_output_dim', 6))
            use_layer_norm = bool(payload.get('mlp_use_layer_norm', True))

            if 'feature_mean' in payload and 'feature_scale' in payload:
                self._feature_mean = torch.tensor(
                    payload['feature_mean'], dtype=torch.float32).reshape(1, -1)
                self._feature_scale = torch.tensor(
                    payload['feature_scale'], dtype=torch.float32).reshape(1, -1)

            if 'loss_motion_error_scale' in payload:
                self._motion_error_clip = 3.0 * torch.tensor(
                    payload['loss_motion_error_scale'],
                    dtype=torch.float32).reshape(1, -1)
        else:
            state_dict = payload if isinstance(payload, dict) else {}
            weights = [v for k, v in state_dict.items()
                       if k.endswith('.weight') and v.ndim == 2]
            if not weights:
                raise ValueError("Checkpoint 不包含线性层权重")
            input_dim = int(weights[0].shape[1])
            output_dim = int(weights[-1].shape[0])
            use_layer_norm = True

        self._mlp = MLPErrorModel(
            input_dim=input_dim, output_dim=output_dim,
            use_layer_norm=use_layer_norm)
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()

        # 冻结 MLP 权重：不参与控制器调参
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

        # 控制量：方向盘角(rad) + 4 轮扭矩(N·m)，后驱两轮平分
        delta_sw = delta * self._steer_ratio
        torque_rear = torque_wheel / 2.0
        zero = torch.zeros_like(torque_wheel)
        control = torch.stack(
            [delta_sw, zero, zero, torque_rear, torque_rear]).unsqueeze(0)

        state = self._state.unsqueeze(0)
        trailer_mass = state.new_tensor([[self._trailer_mass]])
        dt_t = state.new_tensor([[self.dt]])

        # Base RK4 积分（注意：底层 forward 已包含包角）
        base_next = self.dynamics(state, control, trailer_mass, dt_t)

        if self._mlp is not None:
            features = build_mlp_input_feature_tensor(
                state, control, trailer_mass, dt_t)
            if self._feature_mean is not None:
                features = (features - self._feature_mean) / self._feature_scale

            motion_error = self._mlp(features)
            if self._motion_error_clip is not None:
                motion_error = torch.clamp(
                    motion_error,
                    -self._motion_error_clip,
                    self._motion_error_clip)

            full_error = derive_full_error_from_motion_error_torch(
                motion_error, base_next, dt_t)
            self._state = (base_next + full_error).squeeze(0)
            # 修正后再包一次角度
            self._state = self._state.clone()
            self._state[2] = wrap_angle_error_torch(self._state[2:3])[0]
            self._state[8] = wrap_angle_error_torch(self._state[8:9])[0]
        else:
            self._state = base_next.squeeze(0)

    def detach_state(self):
        """截断梯度链（Truncated BPTT）。"""
        self._state = self._state.detach().requires_grad_(False)

    @property
    def x(self):
        """对外暴露牵引车后轴 x 坐标（控制器约定）。"""
        x_cg = self._state[0]
        yaw = self._state[2]
        return x_cg - self._b_t * torch.cos(yaw)

    @property
    def y(self):
        """对外暴露牵引车后轴 y 坐标。"""
        y_cg = self._state[1]
        yaw = self._state[2]
        return y_cg - self._b_t * torch.sin(yaw)

    @property
    def yaw(self):
        """牵引车朝向（前后轴同朝向）。"""
        return self._state[2]

    @property
    def v(self):
        """牵引车后轴速度模 = sqrt(vx² + (vy − b_t·r)²)。"""
        vx = self._state[3]
        vy = self._state[4]
        r = self._state[5]
        vy_rear = vy - self._b_t * r
        return torch.sqrt(vx * vx + vy_rear * vy_rear + 1e-10)

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / math.pi)

    @property
    def trailer_state(self):
        """挂车 6D 状态 [x_s, y_s, psi_s, vx_s, vy_s, r_s]，供可视化记录。"""
        return self._state[6:12]

    @property
    def articulation_rad(self):
        """铰接角 (psi_s − psi_t)，包到 [-π, π]。"""
        return wrap_angle_error_torch(
            (self._state[8] - self._state[2]).unsqueeze(0))[0]
