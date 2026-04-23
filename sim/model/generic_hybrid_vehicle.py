# sim/model/generic_hybrid_vehicle.py
"""通用混合车辆：任意 base 动力学 + checkpoint 驱动的 MLP 残差修正。

特点：
- Base 动力学模型可插拔（通过 base_model_class 参数选择）
- MLP 结构从 checkpoint 元数据自动重建（层数/激活函数/LayerNorm/输入特征）
- 前轴↔后轴坐标转换在内部完成，对外统一暴露后轴坐标
- 接口与 BicycleModel / DynamicVehicle 一致：step(delta, acc), x, y, yaw, v

配置示例 (default.yaml):
  vehicle:
    model_type: hybrid_v2
    base_model: dynamic_v2
    params_section: dynamic_v2_vehicle
    checkpoint_path: configs/checkpoints/best_error_model.pth
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MLP 自动构建
# ---------------------------------------------------------------------------

def _infer_mlp_structure(state_dict):
    """从 state_dict 推断 MLP 结构：输入维度、输出维度、隐层数。"""
    linear_weights = [(k, v) for k, v in state_dict.items()
                      if k.endswith('.weight') and v.ndim == 2]
    if not linear_weights:
        raise ValueError("Checkpoint 不包含线性层权重")
    input_dim = int(linear_weights[0][1].shape[1])
    output_dim = int(linear_weights[-1][1].shape[0])
    n_hidden = len(linear_weights) - 1
    # 隐层宽度取第一个隐层
    hidden_dim = int(linear_weights[0][1].shape[0]) if linear_weights else 128
    return input_dim, output_dim, n_hidden, hidden_dim


def _infer_layer_norm(state_dict):
    """检测是否使用 LayerNorm（通过检查 network.1 是否有 weight/bias）。"""
    return ('network.1.weight' in state_dict and
            'network.1.bias' in state_dict)


class GenericMLPErrorModel(nn.Module):
    """从 checkpoint 元数据自动构建的 MLP 残差网络。"""

    def __init__(self, input_dim, output_dim, n_hidden=4, hidden_dim=128,
                 use_layer_norm=False, dropout_p=0.02):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            else:
                layers.append(nn.Identity())
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_p))
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # 输出层零初始化
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Checkpoint 解析
# ---------------------------------------------------------------------------

def _split_checkpoint(payload):
    """拆分 checkpoint：state_dict + 元数据。"""
    if isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
        metadata = {}
        for key in ('feature_mean', 'feature_scale', 'motion_error_scale',
                    'model_input_dim', 'model_output_dim',
                    'mlp_use_layer_norm', 'input_feature_names'):
            if key in payload:
                metadata[key] = payload[key]
        return state_dict, metadata
    if isinstance(payload, dict):
        return payload, {}
    raise TypeError(f"不支持的 checkpoint 类型: {type(payload)}")


# ---------------------------------------------------------------------------
# 特征构造
# ---------------------------------------------------------------------------

# 10D legacy 特征名（旧版 checkpoint，raw yaw）
_LEGACY_FEATURE_NAMES = ['x', 'y', 'yaw', 'vx', 'vy', 'r',
                         'delta', 'T_rl', 'T_rr', 'dt']
# 11D 新版特征名（sin/cos yaw 分解）
_NEW_FEATURE_NAMES = ['x', 'y', 'sin_yaw', 'cos_yaw', 'vx', 'vy', 'r',
                      'delta', 'T_rl', 'T_rr', 'dt']

# MLP 选用的控制量索引：[delta_sw, T_rl, T_rr] -> control[:, [0, 3, 4]]
_MLP_CONTROL_INDICES = [0, 3, 4]


def _resolve_feature_names(metadata, input_dim):
    """从 metadata 或输入维度推断特征名列表。"""
    raw = metadata.get('input_feature_names')
    if raw is not None:
        names = [str(n) for n in np.asarray(raw).reshape(-1).tolist()]
    elif input_dim == len(_NEW_FEATURE_NAMES):
        names = list(_NEW_FEATURE_NAMES)
    elif input_dim == len(_LEGACY_FEATURE_NAMES):
        names = list(_LEGACY_FEATURE_NAMES)
    else:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} 不匹配已知特征布局 "
            f"(legacy={len(_LEGACY_FEATURE_NAMES)}, new={len(_NEW_FEATURE_NAMES)})")
    if len(names) != input_dim:
        raise ValueError(
            f"特征名数量 ({len(names)}) != input_dim ({input_dim})")
    return names


def _build_feature_map(state, selected_control, dt_tensor):
    """构建特征名→tensor 的映射表。"""
    yaw = state[:, 2:3]
    return {
        'x': state[:, 0:1],
        'y': state[:, 1:2],
        'yaw': yaw,
        'sin_yaw': torch.sin(yaw),
        'cos_yaw': torch.cos(yaw),
        'vx': state[:, 3:4],
        'vy': state[:, 4:5],
        'r': state[:, 5:6],
        'delta': selected_control[:, 0:1],
        'T_rl': selected_control[:, 1:2],
        'T_rr': selected_control[:, 2:3],
        'dt': dt_tensor,
    }


# ---------------------------------------------------------------------------
# 残差重建
# ---------------------------------------------------------------------------

def _reconstruct_full_error(motion_error, base_next, dt):
    """3D 运动残差 [dvx, dvy, dr] → 6D 状态修正 [dx, dy, dyaw, dvx, dvy, dr]。"""
    safe_dt = max(dt, 1.0e-6)
    yaw_ref = base_next[:, 2:3]

    vx_err = motion_error[:, 0:1]
    vy_err = motion_error[:, 1:2]
    r_err = motion_error[:, 2:3]

    cos_yaw = torch.cos(yaw_ref)
    sin_yaw = torch.sin(yaw_ref)

    wx_err = cos_yaw * vx_err - sin_yaw * vy_err
    wy_err = sin_yaw * vx_err + cos_yaw * vy_err

    x_err = wx_err * safe_dt
    y_err = wy_err * safe_dt
    yaw_err = r_err * safe_dt

    pose_err = torch.cat([x_err, y_err, yaw_err], dim=1)
    return torch.cat([pose_err, motion_error], dim=1)


# ---------------------------------------------------------------------------
# GenericHybridVehicle
# ---------------------------------------------------------------------------

class GenericHybridVehicle:
    """通用混合车辆：任意 base 动力学 + 可选 MLP 残差修正。

    内部状态 [x_f, y_f, yaw, vx_f, vy_f, r] 为前轴参考点，
    对外属性 x, y 暴露后轴坐标（自动转换）。

    Args:
        params: 车辆参数字典
        base_model_class: base 动力学模型类（需实现 forward(state, control, dt)→next_state）
        checkpoint_path: MLP checkpoint 路径（None 则纯 base 模型）
    """

    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0,
                 dt=0.02, differentiable=False,
                 checkpoint_path=None, base_model_class=None):
        self.params = params
        self.dt = dt
        self.differentiable = differentiable

        # 延迟导入，避免循环依赖
        if base_model_class is None:
            from model.dynamic_vehicle_v2 import VehicleDynamicsV2
            base_model_class = VehicleDynamicsV2

        self.dynamics = base_model_class(params)
        self._steer_ratio = float(params.get('steer_ratio', 16.39))

        # 输入后轴坐标 → 内部前轴坐标
        L = float(params['lf']) + float(params['lr'])
        self._L = L
        yaw_f = float(yaw)
        x_f = float(x) + L * math.cos(yaw_f)
        y_f = float(y) + L * math.sin(yaw_f)

        self._state = torch.tensor(
            [x_f, y_f, yaw_f, float(v), 0.0, 0.0])

        # MLP 相关（checkpoint 驱动）
        self._mlp = None
        self._feature_names = None
        self._feature_mean = None
        self._feature_scale = None
        self._motion_error_clip = None

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path):
        """加载 MLP checkpoint，自动重建网络结构。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MLP checkpoint 不存在: {path}")

        payload = torch.load(path, map_location='cpu', weights_only=False)
        state_dict, metadata = _split_checkpoint(payload)

        # 推断网络结构
        inferred_in, inferred_out, n_hidden, hidden_dim = \
            _infer_mlp_structure(state_dict)
        input_dim = int(metadata.get('model_input_dim', inferred_in))
        output_dim = int(metadata.get('model_output_dim', inferred_out))
        use_layer_norm = bool(metadata.get(
            'mlp_use_layer_norm', _infer_layer_norm(state_dict)))

        # 特征名
        self._feature_names = _resolve_feature_names(metadata, input_dim)

        # 构建 MLP
        self._mlp = GenericMLPErrorModel(
            input_dim=input_dim, output_dim=output_dim,
            n_hidden=n_hidden, hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm)
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()

        # 冻结权重
        for p in self._mlp.parameters():
            p.requires_grad_(False)

        # 归一化统计量
        if 'feature_mean' in metadata and 'feature_scale' in metadata:
            self._feature_mean = torch.tensor(
                np.asarray(metadata['feature_mean'], dtype=np.float32)
            ).reshape(1, -1)
            self._feature_scale = torch.tensor(
                np.asarray(metadata['feature_scale'], dtype=np.float32)
            ).reshape(1, -1)

        # 残差 clip
        if 'motion_error_scale' in metadata:
            self._motion_error_clip = 3.0 * torch.tensor(
                np.asarray(metadata['motion_error_scale'], dtype=np.float32)
            ).reshape(1, -1)

    def step(self, delta, torque_wheel):
        """前进一步。

        Args:
            delta: 前轮转角 (rad) — 控制器输出
            torque_wheel: 车轮总扭矩 (N·m) — 后驱两轮平分
        """
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(float(delta))
        if not isinstance(torque_wheel, torch.Tensor):
            torque_wheel = torch.tensor(float(torque_wheel))

        # 控制量转换：前轮角→方向盘角，后驱两轮平分总扭矩
        delta_sw = delta * self._steer_ratio
        torque_rear = torque_wheel / 2.0
        zero = torch.zeros_like(torque_wheel)

        control = torch.stack(
            [delta_sw, zero, zero, torque_rear, torque_rear]).unsqueeze(0)
        state = self._state.unsqueeze(0)
        dt_tensor = state.new_tensor([[self.dt]])

        # Base 动力学积分
        base_next = self.dynamics(state, control, dt_tensor)

        if self._mlp is not None:
            # 构造 MLP 特征
            selected_ctrl = control[:, _MLP_CONTROL_INDICES]
            feature_map = _build_feature_map(state, selected_ctrl, dt_tensor)
            features = torch.cat(
                [feature_map[name] for name in self._feature_names], dim=1)

            # 归一化
            if self._feature_mean is not None:
                features = ((features - self._feature_mean)
                            / self._feature_scale)

            # MLP 前向
            motion_error = self._mlp(features)

            # clip
            if self._motion_error_clip is not None:
                motion_error = torch.clamp(
                    motion_error,
                    -self._motion_error_clip,
                    self._motion_error_clip)

            # 重建 6D 修正
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
        return x_f - self._L * torch.cos(yaw)

    @property
    def y(self):
        """后轴 y 坐标（从前轴内部状态转换）。"""
        y_f = self._state[1]
        yaw = self._state[2]
        return y_f - self._L * torch.sin(yaw)

    @property
    def yaw(self):
        return self._state[2]

    @property
    def v(self):
        """合速度 = sqrt(vx² + vy²)。"""
        vx = self._state[3]
        vy = self._state[4]
        return torch.sqrt(vx * vx + vy * vy + 1e-10)

    @property
    def yawrate(self):
        """当前横摆角速度 r (rad/s)，base 动力学 + MLP 残差修正后的值。"""
        return self._state[5]

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / math.pi)
