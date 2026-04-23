# sim/optim/train_batch.py
"""批量并行版可微训练（仅支持 truck_trailer plant）。

与 optim/train.py 等价，但把 48 条轨迹的 50Hz 闭环仿真改成**同步推进**：
每一时间步所有 batch 元素一起过查表 / RK4 / MLP，借助 BLAS 多线程 + 摊薄
Python dispatch 开销，目标单 epoch 从 ~40 min 压到 ~3-5 min。

scalar 路径 (optim/train.py) 保留不动，专用于 post_training 的 V1 验证。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common import normalize_angle  # 支持 [B] tensor
from config import (apply_plant_override, load_config, save_tuned_config,
                    table_from_config)
from controller.lat_truck import LatControllerTruck  # 复用参数导出
from controller.lon import LonController
from model.trajectory import (SPEED_BANDS_KPH, TRAJECTORY_TYPES,
                              expand_trajectories)
from model.truck_trailer_dynamics import (NO_TRAILER_MASS_THRESHOLD_KG,
                                          MLPErrorModel,
                                          TruckTrailerNominalDynamics,
                                          wrap_angle_error_torch)
from model.truck_trailer_vehicle import (
    _resolve_checkpoint_path, build_mlp_input_feature_tensor,
    build_mlp_input_feature_tensor_v2,
    derive_full_error_from_motion_error_torch,
    derive_full_error_from_motion_error_torch_v2)
from model.vehicle_factory import resolve_vehicle_geometry
from optim.train import DiffControllerParams  # 复用 to_config_dict

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ─────────────────────────────────────────────────────────────────────────
# 1. 批量基础算子
# ─────────────────────────────────────────────────────────────────────────

def _lookup1d_batch(table_x: torch.Tensor, table_y: torch.Tensor,
                    x: torch.Tensor) -> torch.Tensor:
    """分段线性插值，边界 clamp。x: [B] → [B]。梯度流向 table_y。"""
    if x.dim() == 0:
        x = x.unsqueeze(0)
    if len(table_x) == 1:
        return table_y[0].expand(x.shape[0]).clone()
    x_clamped = torch.clamp(x, table_x[0].item(), table_x[-1].item())
    idx = torch.searchsorted(table_x, x_clamped) - 1
    idx = torch.clamp(idx, 0, len(table_x) - 2).long()
    x0 = table_x[idx]
    x1 = table_x[idx + 1]
    y0 = table_y[idx]
    y1 = table_y[idx + 1]
    t = (x_clamped - x0) / (x1 - x0 + 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    return y0 + (y1 - y0) * t


class _StraightThroughClampBatch(torch.autograd.Function):
    """Forward: 硬 clamp；Backward: 对 x 直通，lo/hi 仅在饱和侧收梯度。支持 [B]。"""

    @staticmethod
    def forward(ctx, x, lo, hi):
        lo_t = lo if isinstance(lo, torch.Tensor) else torch.as_tensor(
            float(lo), dtype=x.dtype)
        hi_t = hi if isinstance(hi, torch.Tensor) else torch.as_tensor(
            float(hi), dtype=x.dtype)
        lo_b = lo_t.expand_as(x)
        hi_b = hi_t.expand_as(x)
        ctx.save_for_backward(x, lo_b, hi_b)
        ctx.lo_is_t = isinstance(lo, torch.Tensor)
        ctx.hi_is_t = isinstance(hi, torch.Tensor)
        return torch.maximum(torch.minimum(x, hi_b), lo_b)

    @staticmethod
    def backward(ctx, grad_out):
        x, lo_b, hi_b = ctx.saved_tensors
        grad_x = grad_out
        grad_lo = (grad_out * (x <= lo_b).to(grad_out.dtype)
                   if ctx.lo_is_t else None)
        grad_hi = (grad_out * (x >= hi_b).to(grad_out.dtype)
                   if ctx.hi_is_t else None)
        return grad_x, grad_lo, grad_hi


def _clamp_ste_batch(x: torch.Tensor, lo, hi) -> torch.Tensor:
    return _StraightThroughClampBatch.apply(x, lo, hi)


def _rate_limit_batch(prev: torch.Tensor, target: torch.Tensor,
                      rate: float, dt: float) -> torch.Tensor:
    """前向硬 clamp，反向直通。prev/target 形状一致。"""
    max_delta = float(rate) * float(dt)
    delta = target - prev
    clamped = _clamp_ste_batch(delta, -max_delta, max_delta)
    return prev + clamped


class _PIDBatch:
    """每个 batch 元素独立积分状态。参数 kp/ki/kd 可为标量或 [B]。"""

    def __init__(self, batch_size: int):
        self.B = batch_size
        self.integral = torch.zeros(batch_size)
        self.prev_error = torch.zeros(batch_size)

    def control(self, error, dt, kp, ki, kd, integrator_enable, sat):
        if integrator_enable:
            self.integral = _clamp_ste_batch(
                self.integral + error * dt,
                float(-sat), float(sat))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error.detach().clone()
        return kp * error + ki * self.integral + kd * derivative

    def reset(self):
        self.integral = torch.zeros(self.B)
        self.prev_error = torch.zeros(self.B)

    def detach(self):
        self.integral = self.integral.detach()
        self.prev_error = self.prev_error.detach()


class _IIRBatch:
    """y = x - alpha * y_prev，y_prev 每元素独立。"""

    def __init__(self, alpha, batch_size: int):
        self.alpha = (alpha if isinstance(alpha, torch.Tensor)
                      else torch.tensor(float(alpha)))
        self.B = batch_size
        self.y_prev = torch.zeros(batch_size)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        y = x - self.alpha * self.y_prev
        self.y_prev = y.detach().clone()
        return y

    def reset(self):
        self.y_prev = torch.zeros(self.B)

    def detach(self):
        self.y_prev = self.y_prev.detach()


# ─────────────────────────────────────────────────────────────────────────
# 2. 批量轨迹容器
# ─────────────────────────────────────────────────────────────────────────

class BatchedTrajectoryTable:
    """B 条变长轨迹 padding 到 [B, T_max]，提供批量最近点 / 时间插值查询。

    Padding 位置的 ref 值用末尾值填充（避免 0 带来的误查询），valid_mask
    精确标记每条轨迹的有效步数。
    """

    def __init__(self, trajectories: list):
        self.trajectories = trajectories
        self.B = len(trajectories)
        self.T_max = max(len(t) for t in trajectories)

        def _stack(attr):
            out = torch.zeros((self.B, self.T_max))
            for b, traj in enumerate(trajectories):
                vals = torch.tensor([float(getattr(p, attr)) for p in traj])
                out[b, :len(traj)] = vals
                if len(traj) < self.T_max:
                    out[b, len(traj):] = vals[-1]  # 用末尾值填充 padding
            return out

        self.ref_x = _stack('x')
        self.ref_y = _stack('y')
        self.ref_theta = _stack('theta')
        self.ref_kappa = _stack('kappa')
        self.ref_v = _stack('v')
        self.ref_a = _stack('a')
        self.ref_s = _stack('s')
        self.ref_t = _stack('t')

        mask = torch.zeros((self.B, self.T_max))
        lens = torch.zeros(self.B, dtype=torch.long)
        for b, traj in enumerate(trajectories):
            mask[b, :len(traj)] = 1.0
            lens[b] = len(traj)
        self.valid_mask = mask
        self.lengths = lens

        self.init_x = torch.tensor([float(t[0].x) for t in trajectories])
        self.init_y = torch.tensor([float(t[0].y) for t in trajectories])
        self.init_yaw = torch.tensor([float(t[0].theta) for t in trajectories])
        self.init_v = torch.tensor([float(t[0].v) for t in trajectories])

        # 每条轨迹的 t_max（用于时间查询边界处理）
        self.t_max = torch.tensor([float(t[-1].t) for t in trajectories])

    def query_nearest_idx(self, x: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor:
        """返回每个 batch 元素在其轨迹上的最近点索引（detached）。

        Padding 位置被置为 +inf 以避免被选中。
        """
        with torch.no_grad():
            dx = self.ref_x - x.detach()[:, None]
            dy = self.ref_y - y.detach()[:, None]
            dist2 = dx * dx + dy * dy
            dist2 = torch.where(self.valid_mask > 0.5, dist2,
                                torch.full_like(dist2, 1e18))
            idx = torch.argmin(dist2, dim=1)
        return idx

    def gather_ref(self, idx: torch.Tensor) -> dict:
        """按 [B] idx 取 ref 值，返回 dict of [B]。"""
        b_idx = torch.arange(self.B)
        return {
            'x': self.ref_x[b_idx, idx],
            'y': self.ref_y[b_idx, idx],
            'theta': self.ref_theta[b_idx, idx],
            'kappa': self.ref_kappa[b_idx, idx],
            'v': self.ref_v[b_idx, idx],
            'a': self.ref_a[b_idx, idx],
            's': self.ref_s[b_idx, idx],
            't': self.ref_t[b_idx, idx],
        }

    def query_by_time(self, t_query: torch.Tensor):
        """每个 batch 元素按自己的 t 表做 1D 线性插值（完全向量化）。

        t_query: [B] → (kappa, v, a, s)，每项 [B]。梯度通过线性插值回传到 t。

        padding 步的 ref_t 值用末尾值填充，所以 clamp 到 t_max 后 searchsorted
        落进任一 padding 位置都返回相同末尾值（插值系数退化为 0）。
        """
        t_clamped = torch.minimum(torch.clamp(t_query, min=0.0), self.t_max)
        # 2D searchsorted：self.ref_t [B, T_max], values [B, 1] → [B, 1]
        idx = torch.searchsorted(
            self.ref_t, t_clamped.unsqueeze(1)).squeeze(1) - 1  # [B]
        idx = idx.clamp(0, self.T_max - 2)
        b_idx = torch.arange(self.B)
        t0 = self.ref_t[b_idx, idx]
        t1 = self.ref_t[b_idx, idx + 1]
        frac = ((t_clamped - t0) / (t1 - t0 + 1e-12)).clamp(0.0, 1.0)

        def _interp(arr):
            y0 = arr[b_idx, idx]
            y1 = arr[b_idx, idx + 1]
            return y0 + frac * (y1 - y0)

        return (_interp(self.ref_kappa), _interp(self.ref_v),
                _interp(self.ref_a), _interp(self.ref_s))


# ─────────────────────────────────────────────────────────────────────────
# 3. 批量 truck_trailer 车辆（直接复用已有的批量动力学/MLP 模块）
# ─────────────────────────────────────────────────────────────────────────

class BatchedTruckTrailerVehicle:
    """TruckTrailerNominalDynamics.forward 已接受 [B,12] 状态，这里只做包装。

    对外暴露 x/y/yaw/v 为牵引车后轴（匹配 sim 控制器约定）。
    """

    def __init__(self, cfg: dict, batch_size: int,
                 init_x: torch.Tensor, init_y: torch.Tensor,
                 init_yaw: torch.Tensor, init_v: torch.Tensor,
                 dt: float = 0.02, trailer_mass_kg=None,
                 checkpoint_path=None):
        self.B = batch_size
        params = cfg['truck_trailer_vehicle']
        self.dt = dt
        self.dynamics = TruckTrailerNominalDynamics(params)

        self._steer_ratio = float(params['steering_ratio'])
        self._L_t = float(params['L_t'])
        self._a_t = float(params['a_t'])
        self._b_t = self._L_t - self._a_t

        if trailer_mass_kg is None:
            default_m = float(params.get('default_trailer_mass_kg',
                                          params['m_s_base']))
            trailer_mass_kg = torch.full((batch_size,), default_m)
        else:
            trailer_mass_kg = trailer_mass_kg.to(torch.float32).view(batch_size)
        self._trailer_mass = trailer_mass_kg

        # 后轴 → 质心：x_cg = x_rear + b_t * cos(yaw)
        yaw_f = init_yaw.to(torch.float32).view(batch_size)
        v_f = init_v.to(torch.float32).view(batch_size)
        x_rear = init_x.to(torch.float32).view(batch_size)
        y_rear = init_y.to(torch.float32).view(batch_size)
        x_t_cg = x_rear + self._b_t * torch.cos(yaw_f)
        y_t_cg = y_rear + self._b_t * torch.sin(yaw_f)
        c_s = float(params['c_s'])
        hx = float(params['hitch_x'])
        hy = float(params['hitch_y'])
        x_hitch = x_t_cg + hx * torch.cos(yaw_f) - hy * torch.sin(yaw_f)
        y_hitch = y_t_cg + hx * torch.sin(yaw_f) + hy * torch.cos(yaw_f)
        x_s_cg = x_hitch - c_s * torch.cos(yaw_f)
        y_s_cg = y_hitch - c_s * torch.sin(yaw_f)
        zeros = torch.zeros(batch_size)
        self._state = torch.stack([
            x_t_cg, y_t_cg, yaw_f, v_f, zeros, zeros,
            x_s_cg, y_s_cg, yaw_f, v_f, zeros, zeros,
        ], dim=1)  # [B, 12]

        self._mlp = None
        self._feature_mean = None
        self._feature_scale = None
        self._motion_error_clip = None
        self._mlp_input_dim = None
        self._mlp_output_dim = None

        ckpt = _resolve_checkpoint_path(
            checkpoint_path or params.get('checkpoint_path'))
        if ckpt:
            self._load_checkpoint(ckpt)

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        hidden_dim, hidden_layers = 128, 4
        if isinstance(payload, dict) and 'state_dict' in payload:
            sd = payload['state_dict']
            input_dim = int(payload.get('model_input_dim', 18))
            output_dim = int(payload.get('model_output_dim', 6))
            use_ln = bool(payload.get('mlp_use_layer_norm', True))
            hidden_dim = int(payload.get('mlp_hidden_dim', hidden_dim))
            hidden_layers = int(payload.get('mlp_hidden_layers', hidden_layers))
            if 'feature_mean' in payload:
                self._feature_mean = torch.tensor(
                    payload['feature_mean'],
                    dtype=torch.float32).reshape(1, -1)
                self._feature_scale = torch.tensor(
                    payload['feature_scale'],
                    dtype=torch.float32).reshape(1, -1)
            if 'loss_output_scale' in payload and output_dim == 9:
                self._motion_error_clip = 3.0 * torch.tensor(
                    payload['loss_output_scale'],
                    dtype=torch.float32).reshape(1, -1)
            elif 'loss_motion_error_scale' in payload:
                self._motion_error_clip = 3.0 * torch.tensor(
                    payload['loss_motion_error_scale'],
                    dtype=torch.float32).reshape(1, -1)
        else:
            sd = payload
            weights = [v for k, v in sd.items()
                       if k.endswith('.weight') and v.ndim == 2]
            input_dim = int(weights[0].shape[1])
            output_dim = int(weights[-1].shape[0])
            use_ln = True
            hidden_dim = int(weights[0].shape[0])
            hidden_layers = max(1, len(weights) - 1)
        self._mlp = MLPErrorModel(
            input_dim=input_dim, output_dim=output_dim,
            use_layer_norm=use_ln,
            hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self._mlp.load_state_dict(sd)
        self._mlp.eval()
        self._mlp_input_dim = input_dim
        self._mlp_output_dim = output_dim
        for p in self._mlp.parameters():
            p.requires_grad_(False)

    def step(self, delta: torch.Tensor, torque_wheel: torch.Tensor):
        """delta/torque_wheel: [B]。delta 前轮转角 (rad)，torque 车轮总扭矩。"""
        delta_sw = delta * self._steer_ratio
        torque_rear = torque_wheel / 2.0
        zero = torch.zeros_like(torque_wheel)
        control = torch.stack(
            [delta_sw, zero, zero, torque_rear, torque_rear], dim=1)  # [B,5]

        trailer_mass = self._trailer_mass.unsqueeze(1)  # [B,1]
        dt_t = torch.full((self.B, 1), self.dt)

        base_next = self.dynamics(self._state, control, trailer_mass, dt_t)

        if self._mlp is not None:
            if self._mlp_input_dim == 14:
                features = build_mlp_input_feature_tensor_v2(
                    self._state, control, trailer_mass, dt_t)
            else:
                features = build_mlp_input_feature_tensor(
                    self._state, control, trailer_mass, dt_t)
            if self._feature_mean is not None:
                features = (features - self._feature_mean) / self._feature_scale
            motion_error = self._mlp(features)
            if self._motion_error_clip is not None:
                motion_error = torch.clamp(
                    motion_error,
                    -self._motion_error_clip, self._motion_error_clip)
            if self._mlp_output_dim == 9:
                full_error = derive_full_error_from_motion_error_torch_v2(
                    motion_error, base_next, dt_t)
            else:
                full_error = derive_full_error_from_motion_error_torch(
                    motion_error, base_next, dt_t)
            new_state = base_next + full_error
            new_state = new_state.clone()
            new_state[:, 2] = wrap_angle_error_torch(new_state[:, 2])
            new_state[:, 8] = wrap_angle_error_torch(new_state[:, 8])
            # 无挂车：强制挂车态=牵引车态（MLP 修正后可能破坏该不变量）
            no_tr = (self._trailer_mass <= NO_TRAILER_MASS_THRESHOLD_KG)
            if no_tr.any():
                mask = no_tr.view(-1, 1).expand(-1, 6)
                new_state = torch.cat([
                    new_state[:, 0:6],
                    torch.where(mask, new_state[:, 0:6], new_state[:, 6:12]),
                ], dim=1)
            self._state = new_state
        else:
            self._state = base_next

    def detach_state(self):
        self._state = self._state.detach()

    @property
    def x(self):
        return self._state[:, 0] - self._b_t * torch.cos(self._state[:, 2])

    @property
    def y(self):
        return self._state[:, 1] - self._b_t * torch.sin(self._state[:, 2])

    @property
    def yaw(self):
        return self._state[:, 2]

    @property
    def v(self):
        vx = self._state[:, 3]
        vy = self._state[:, 4]
        r = self._state[:, 5]
        vy_rear = vy - self._b_t * r
        return torch.sqrt(vx * vx + vy_rear * vy_rear + 1e-10)

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * RAD2DEG


# ─────────────────────────────────────────────────────────────────────────
# 4. 批量横向 + 纵向控制器
# ─────────────────────────────────────────────────────────────────────────

# smooth_* 的批量友好版（原始版对 [B] 输入会在 half.item() 处炸）。
def _smooth_lower_bound(x, lo, sharpness=10.0):
    lo_val = float(lo) if not isinstance(lo, torch.Tensor) else lo
    return lo_val + torch.nn.functional.softplus(
        (x - lo_val) * sharpness, beta=1.0) / sharpness


def _smooth_upper_bound(x, hi, sharpness=10.0):
    hi_val = float(hi) if not isinstance(hi, torch.Tensor) else hi
    return hi_val - torch.nn.functional.softplus(
        (hi_val - x) * sharpness, beta=1.0) / sharpness


def _smooth_step(x, threshold, temp=1.0):
    th = float(threshold) if not isinstance(threshold, torch.Tensor) else threshold
    return torch.sigmoid((x - th) / temp)


def _smooth_clamp_batch(x, lo, hi, temp=1.0):
    """lo/hi 可为 [B] 张量的 smooth_clamp。"""
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return mid + half * torch.tanh((x - mid) / (half * temp + 1e-12))


# hard_mode 对偶：语义和 V1（differentiable=False）路径完全一致的硬限幅/硬阶跃。
# 用于 V1 验证场景（不需要梯度，要求与 scalar run_simulation 数值严格一致）。
def _hard_lower_bound(x, lo, **_):
    lo_t = lo if isinstance(lo, torch.Tensor) else torch.as_tensor(
        float(lo), dtype=x.dtype)
    return torch.maximum(x, lo_t)


def _hard_upper_bound(x, hi, **_):
    hi_t = hi if isinstance(hi, torch.Tensor) else torch.as_tensor(
        float(hi), dtype=x.dtype)
    return torch.minimum(x, hi_t)


def _hard_step(x, threshold, **_):
    th = threshold if isinstance(threshold, torch.Tensor) else torch.as_tensor(
        float(threshold), dtype=x.dtype)
    # scalar 路径是严格 > 判断（非 smooth），硬模式下用 > 和 float cast 复刻
    return (x > th).to(x.dtype)


def _hard_clamp_batch(x, lo, hi, **_):
    lo_t = lo if isinstance(lo, torch.Tensor) else torch.as_tensor(
        float(lo), dtype=x.dtype)
    hi_t = hi if isinstance(hi, torch.Tensor) else torch.as_tensor(
        float(hi), dtype=x.dtype)
    return torch.minimum(torch.maximum(x, lo_t), hi_t)


def _pick_ops(hard_mode: bool):
    """返回控制器内部用的 5 个 op（lower_bound/upper_bound/step/clamp/ste_clamp）。
    hard_mode=False 用 smooth 版（训练可梯度），True 用硬限幅（V1 验证语义）。
    rate_limit / STE clamp 的 forward 本就是硬 clamp，差别仅在反向；hard 模式下
    走同一路径但套 no_grad 即可（调用方负责）。"""
    if hard_mode:
        return (_hard_lower_bound, _hard_upper_bound, _hard_step,
                _hard_clamp_batch)
    return (_smooth_lower_bound, _smooth_upper_bound, _smooth_step,
            _smooth_clamp_batch)


class BatchedLatTruck(nn.Module):
    """横向控制器批量版。nn.Parameter 名/shape 与 scalar LatControllerTruck
    完全一致（T2/T3/T4/T6 查找表 y 值），方便直接 copy_ 过去做参数导出。"""

    def __init__(self, cfg: dict, batch_size: int):
        super().__init__()
        lat = cfg['lat_truck']
        self.wheelbase, self.steer_ratio = resolve_vehicle_geometry(cfg)
        self.B = batch_size

        self.register_buffer('kLh', torch.tensor(float(lat['kLh'])))
        self.rate_limit_fb = lat['rate_limit_fb']
        self.rate_limit_ff = lat['rate_limit_ff']
        self.rate_limit_total = lat['rate_limit_total']
        self.min_prev_dist = lat['min_prev_dist']
        self.min_reach_dis = lat['min_reach_dis']
        self.min_speed_prot = lat['min_speed_prot']

        _fixed = {'T1', 'T5', 'T7', 'T8'}
        key_map = {
            'T1': 'T1_max_theta_deg', 'T2': 'T2_prev_time_dist',
            'T3': 'T3_reach_time_theta', 'T4': 'T4_T_dt',
            'T5': 'T5_near_point_time', 'T6': 'T6_far_point_time',
            'T7': 'T7_max_steer_angle', 'T8': 'T8_slip_param',
        }
        for name in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            xs, ys = table_from_config(lat[key_map[name]])
            self.register_buffer(f'{name}_x', xs)
            if name in _fixed:
                self.register_buffer(f'{name}_y', ys)
            else:
                setattr(self, f'{name}_y', nn.Parameter(ys))

        self.register_buffer('steer_fb_prev', torch.zeros(batch_size))
        self.register_buffer('steer_ff_prev', torch.zeros(batch_size))
        self.register_buffer('steer_total_prev', torch.zeros(batch_size))

    def reset_state(self):
        self.steer_fb_prev.zero_()
        self.steer_ff_prev.zero_()
        self.steer_total_prev.zero_()

    def detach_state(self):
        self.steer_fb_prev = self.steer_fb_prev.detach()
        self.steer_ff_prev = self.steer_ff_prev.detach()
        self.steer_total_prev = self.steer_total_prev.detach()

    def compute(self, x, y, yaw_deg, speed_kph, yawrate, steer_feedback,
                btraj: BatchedTrajectoryTable, dt: float = 0.02,
                hard_mode: bool = False):
        """输入全部 [B]。返回 (steer_out, kappa_cur, near_kappa, far_kappa,
        steer_fb, steer_ff) 每项 [B]。
        hard_mode=True 时走硬限幅路径（与 V1 scalar run_simulation 等价），
        False 时走 smooth 近似（训练/梯度路径）。"""
        lb, _ub, _step, clamp = _pick_ops(hard_mode)
        speed_kph_safe = lb(speed_kph, self.min_speed_prot)
        speed_mps = speed_kph_safe / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        # Step 1: 8 张查表（结果 [B]，参数 y 可梯度）
        max_theta_deg = _lookup1d_batch(self.T1_x, self.T1_y, speed_kph_safe)
        prev_time_dist = _lookup1d_batch(self.T2_x, self.T2_y, speed_kph_safe)
        reach_time_theta = _lookup1d_batch(self.T3_x, self.T3_y, speed_kph_safe)
        T_dt = _lookup1d_batch(self.T4_x, self.T4_y, speed_kph_safe)
        near_pt_time = _lookup1d_batch(self.T5_x, self.T5_y, speed_kph_safe)
        far_pt_time = _lookup1d_batch(self.T6_x, self.T6_y, speed_kph_safe)
        max_steer_angle = _lookup1d_batch(self.T7_x, self.T7_y, speed_kph_safe)
        slip_param = _lookup1d_batch(self.T8_x, self.T8_y, speed_kph_safe)

        # Step 2: 最近点 + 时间预瞄
        idx_curr = btraj.query_nearest_idx(x, y)
        ref_curr = btraj.gather_ref(idx_curr)
        t_base = ref_curr['t']
        near_kappa, _, _, _ = btraj.query_by_time(t_base + near_pt_time)
        far_kappa, _, _, _ = btraj.query_by_time(t_base + far_pt_time)

        # Step 3: 误差
        dx = x - ref_curr['x']
        dy = y - ref_curr['y']
        cos_th = torch.cos(ref_curr['theta'])
        sin_th = torch.sin(ref_curr['theta'])
        lateral_error = cos_th * dy - sin_th * dx
        heading_error = normalize_angle(yaw_rad - ref_curr['theta'])
        curvature_far = far_kappa

        # Step 4
        v_clamped = clamp(
            speed_mps,
            torch.tensor(1.0).expand_as(speed_mps),
            torch.tensor(100.0).expand_as(speed_mps), temp=1.0)
        real_theta = heading_error + torch.atan(
            self.kLh * yawrate / v_clamped)

        # Step 5
        real_dt_theta = yawrate - curvature_far * speed_mps

        # Step 6: target_theta
        prev_dist = lb(speed_mps * prev_time_dist, self.min_prev_dist)
        dis2lane = -lateral_error
        error_angle_raw = torch.atan(dis2lane / prev_dist)
        max_err_limit = max_theta_deg * DEG2RAD
        # STE clamp 的 forward 与硬 clamp 等价，直接复用（no_grad 下无反向开销）
        target_theta = _clamp_ste_batch(
            error_angle_raw, -max_err_limit, max_err_limit)

        target_dt_theta = (torch.sin(real_theta) * speed_mps * prev_dist
                           / (prev_dist ** 2 + dis2lane ** 2) * -1.0)

        # Step 7
        denom = lb(reach_time_theta * speed_mps, self.min_reach_dis)
        target_curvature = ((target_theta - real_theta)
                            + (target_dt_theta - real_dt_theta) * T_dt) / denom

        # Step 8: feedback
        steer_fb_raw = (torch.atan(target_curvature * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_fb = _rate_limit_batch(
            self.steer_fb_prev, steer_fb_raw, self.rate_limit_fb, dt)
        self.steer_fb_prev = steer_fb.detach().clone()

        # Step 9: feedforward
        steer_ff_raw = (torch.atan(curvature_far * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_ff = _rate_limit_batch(
            self.steer_ff_prev, steer_ff_raw, self.rate_limit_ff, dt)
        self.steer_ff_prev = steer_ff.detach().clone()

        # Step 10: 合并
        steer_raw = clamp(
            steer_fb + steer_ff, -max_steer_angle, max_steer_angle, temp=1.0)
        steer_out = _rate_limit_batch(
            self.steer_total_prev, steer_raw, self.rate_limit_total, dt)
        self.steer_total_prev = steer_out.detach().clone()

        return (steer_out, ref_curr['kappa'], near_kappa, curvature_far,
                steer_fb, steer_ff)


class BatchedLonCtrl(nn.Module):
    """纵向控制器批量版。nn.Parameter 名/shape 与 scalar LonController 完全一致。"""

    def __init__(self, cfg: dict, batch_size: int):
        super().__init__()
        lon = cfg['lon']
        self.B = batch_size

        self.station_kp = nn.Parameter(torch.tensor(float(lon['station_kp'])))
        self.station_ki = nn.Parameter(torch.tensor(float(lon['station_ki'])))
        self.low_speed_kp = nn.Parameter(torch.tensor(float(lon['low_speed_kp'])))
        self.low_speed_ki = nn.Parameter(torch.tensor(float(lon['low_speed_ki'])))
        self.high_speed_kp = nn.Parameter(torch.tensor(float(lon['high_speed_kp'])))
        self.high_speed_ki = nn.Parameter(torch.tensor(float(lon['high_speed_ki'])))
        self.switch_speed = nn.Parameter(torch.tensor(float(lon['switch_speed'])))

        self.preview_window = lon['preview_window']
        self.preview_window_speed = lon['preview_window_speed']
        self.acc_use_preview_a = lon['acc_use_preview_a']
        self.station_error_limit = lon['station_error_limit']
        self.speed_input_limit = lon['speed_input_limit']
        self.acc_standstill_down_rate = lon['acc_standstill_down_rate']
        self.station_integrator_enable = lon['station_integrator_enable']
        self.station_sat = lon['station_sat']
        self.speed_pid_sat = lon['speed_pid_sat']

        for name, key in [('L1', 'L1_acc_up_lim'), ('L2', 'L2_acc_low_lim'),
                           ('L3', 'L3_acc_up_rate'), ('L4', 'L4_acc_down_rate'),
                           ('L5', 'L5_rate_gain')]:
            xs, ys = table_from_config(lon[key])
            self.register_buffer(f'{name}_x', xs)
            self.register_buffer(f'{name}_y', ys)

        self.station_pid = _PIDBatch(batch_size)
        self.speed_pid = _PIDBatch(batch_size)
        self.iir_acc = _IIRBatch(alpha=lon['iir_alpha'], batch_size=batch_size)
        self.register_buffer('acc_out_prev', torch.zeros(batch_size))
        self.register_buffer('station_error_fnl_prev', torch.zeros(batch_size))

        lon_torque = cfg.get('lon_torque', {})
        for key, default in [
            ('veh_mass', 9300.0), ('coef_cd', 0.6),
            ('coef_rolling', 0.013), ('coef_delta', 1.05),
            ('air_density', 1.2041), ('gravity', 9.81),
            ('frontal_area', 9.7), ('wheel_rolling_radius', 0.5),
            ('accel_to_torque_kp', 1000.0), ('accel_deadzone', -0.05),
        ]:
            self.register_buffer(
                f'torque_{key}',
                torch.tensor(float(lon_torque.get(key, default))))

    def reset_state(self):
        self.station_pid.reset()
        self.speed_pid.reset()
        self.iir_acc.reset()
        self.acc_out_prev.zero_()
        self.station_error_fnl_prev.zero_()

    def detach_state(self):
        self.station_pid.detach()
        self.speed_pid.detach()
        self.iir_acc.detach()
        self.acc_out_prev = self.acc_out_prev.detach()
        self.station_error_fnl_prev = self.station_error_fnl_prev.detach()

    def compute(self, x, y, yaw_deg, speed_kph, curvature_far,
                btraj: BatchedTrajectoryTable, t_now,
                ctrl_first_active: bool, dt: float = 0.02,
                hard_mode: bool = False):
        """输入 [B]。返回 acc_cmd [B]。t_now: [B]。
        hard_mode=True 走硬限幅（V1 等价），False 走 smooth 近似（训练）。"""
        _lb, ub, step, _clamp = _pick_ops(hard_mode)
        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD
        if ctrl_first_active:
            self.reset_state()

        # Step 1: Frenet 误差（最近点投影）
        idx_curr = btraj.query_nearest_idx(x, y)
        ref_curr = btraj.gather_ref(idx_curr)
        m_theta = ref_curr['theta']
        proj = ((x - ref_curr['x']) * torch.cos(m_theta)
                + (y - ref_curr['y']) * torch.sin(m_theta))
        s_match = ref_curr['s'] + proj

        _, _, _, ref_s = btraj.query_by_time(t_now)
        _, _, prev_a, _ = btraj.query_by_time(
            t_now + self.preview_window * dt)
        _, spd_v, _, _ = btraj.query_by_time(
            t_now + self.preview_window_speed * dt)

        station_error = ref_s - s_match
        preview_speed_error = spd_v - speed_mps
        preview_accel_ref = prev_a

        # Step 2: 站位误差保护 — 用 torch.where 逐元素精确复刻 scalar 分支：
        #   if speed>10: station_limited
        #   elif station_limited <= 0.25: {smooth|hard} min(station_limited, 0)
        #   elif station_limited >= 0.8: station_limited
        #   elif station_error_fnl_prev <= 0.01: station_error_fnl_prev (stale prev)
        #   else: station_limited
        station_limited = _clamp_ste_batch(
            station_error,
            -self.station_error_limit, self.station_error_limit)
        high_speed = speed_kph > 10.0
        small_pos = station_limited <= 0.25
        large_pos = station_limited >= 0.8
        prev_near_zero = self.station_error_fnl_prev <= 0.01
        low_branch_small = ub(station_limited, 0.0)
        low_station_fnl = torch.where(
            small_pos, low_branch_small,
            torch.where(
                large_pos, station_limited,
                torch.where(prev_near_zero,
                             self.station_error_fnl_prev, station_limited)))
        station_fnl = torch.where(high_speed, station_limited, low_station_fnl)
        self.station_error_fnl_prev = station_fnl.detach().clone()

        # Step 3: 站位 PID
        speed_offset = self.station_pid.control(
            station_fnl, dt,
            kp=self.station_kp, ki=self.station_ki, kd=0.0,
            integrator_enable=self.station_integrator_enable,
            sat=self.station_sat)

        # Step 4: 速度 PID（smooth_step 混合增益；hard 模式下 step 是严格 >）
        w_low = 1.0 - step(speed_mps, self.switch_speed, temp=0.5)
        kp = w_low * self.low_speed_kp + (1.0 - w_low) * self.high_speed_kp
        ki = w_low * self.low_speed_ki + (1.0 - w_low) * self.high_speed_ki

        speed_input_raw = speed_offset + preview_speed_error
        speed_input = _clamp_ste_batch(
            speed_input_raw,
            -self.speed_input_limit, self.speed_input_limit)
        acc_closeloop = self.speed_pid.control(
            speed_input, dt,
            kp=kp, ki=ki, kd=0.0,
            integrator_enable=True, sat=self.speed_pid_sat)

        # Step 5
        acc_cmd = acc_closeloop + self.acc_use_preview_a * preview_accel_ref

        # Step 6: CalFinalAccCmd
        abs_speed_kph = torch.abs(speed_kph)
        acc_up_lim = _lookup1d_batch(self.L1_x, self.L1_y, abs_speed_kph)
        acc_low_lim = _lookup1d_batch(self.L2_x, self.L2_y, abs_speed_kph)
        acc_up_rate_raw = _lookup1d_batch(
            self.L3_x, self.L3_y, self.acc_out_prev)
        acc_dn_rate_raw = _lookup1d_batch(
            self.L4_x, self.L4_y, self.acc_out_prev)
        rate_gain = _lookup1d_batch(self.L5_x, self.L5_y, abs_speed_kph)
        acc_up_rate = acc_up_rate_raw * rate_gain

        w_curv = step(-curvature_far, 0.0075, temp=0.01)
        acc_up_lim_adj = acc_up_lim * (1.0 - 0.25 * w_curv)
        acc_low_lim_adj = acc_low_lim * (1.0 - 0.40 * w_curv)

        abs_speed = torch.abs(speed_mps)
        w_standstill = 1.0 - step(abs_speed, 1.5, temp=0.3)
        acc_dn_rate = (w_standstill * self.acc_standstill_down_rate
                       + (1.0 - w_standstill) * acc_dn_rate_raw)

        acc_clamped = _clamp_ste_batch(acc_cmd, acc_low_lim_adj, acc_up_lim_adj)

        w_pass = step(abs_speed, 0.2, temp=0.05)
        w_acc_ok = step(acc_clamped, 0.25, temp=0.05)
        w_normal = 1.0 - (1.0 - w_pass) * (1.0 - w_acc_ok)
        acc_creep = ub(acc_clamped, -0.05)
        acc_lowspd = w_normal * acc_clamped + (1.0 - w_normal) * acc_creep

        acc_limited = _clamp_ste_batch(
            acc_lowspd,
            self.acc_out_prev + acc_dn_rate,
            self.acc_out_prev + acc_up_rate)
        self.acc_out_prev = acc_limited.detach().clone()

        acc_out = self.iir_acc.update(acc_limited)
        return acc_out

    def compute_torque_wheel(self, acc_cmd, speed_mps, a_actual):
        """[B] → [B]。公式与 scalar _compute_torque_differentiable 一致。"""
        F_air = (0.5 * self.torque_coef_cd * self.torque_air_density
                 * self.torque_frontal_area * speed_mps * speed_mps)
        F_rolling = (self.torque_coef_rolling * self.torque_veh_mass
                     * self.torque_gravity)
        F_inertia = self.torque_coef_delta * self.torque_veh_mass * acc_cmd
        F_resist = F_air + F_rolling + F_inertia
        F_P = self.torque_accel_to_torque_kp * (acc_cmd - a_actual)
        T_raw = (F_resist + F_P) * self.torque_wheel_rolling_radius
        mask = (acc_cmd > self.torque_accel_deadzone).float().detach()
        return mask * T_raw


# ─────────────────────────────────────────────────────────────────────────
# 5. 主循环 / loss / 训练入口
# ─────────────────────────────────────────────────────────────────────────

def run_simulation_batch(trajectories: list, cfg: dict = None,
                         lat_ctrl: BatchedLatTruck = None,
                         lon_ctrl: BatchedLonCtrl = None,
                         tbptt_k: int = 0,
                         hard_mode: bool = False,
                         trailer_mass_kg=None) -> dict:
    """B 条轨迹同步推进 50Hz 闭环。仅支持 truck_trailer plant。

    Args:
        hard_mode: True 时控制器走硬限幅路径（与 V1 scalar run_simulation 等价），
                   整个主循环自动套 torch.no_grad() 省显存/跳过 autograd。
                   False 时走 smooth 近似（训练用，需要梯度回传）。
        trailer_mass_kg: 标量或 [B] tensor，覆盖 cfg 默认挂车质量。

    返回 dict：每项 [B, T_max]，另含 'valid_mask' [B, T_max] 和控制器句柄
    供上层做 loss / 导出用。
    """
    if cfg is None:
        cfg = load_config()
    assert cfg['vehicle'].get('model_type') == 'truck_trailer', \
        "run_simulation_batch 目前仅支持 truck_trailer"

    grad_ctx = torch.no_grad() if hard_mode else _nullctx()
    with grad_ctx:
        return _run_sim_batch_inner(
            trajectories, cfg, lat_ctrl, lon_ctrl, tbptt_k,
            hard_mode, trailer_mass_kg)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _run_sim_batch_inner(trajectories, cfg, lat_ctrl, lon_ctrl, tbptt_k,
                         hard_mode, trailer_mass_override):
    bt = BatchedTrajectoryTable(trajectories)
    B, T_max = bt.B, bt.T_max
    dt = cfg['simulation']['dt']

    if lat_ctrl is None:
        lat_ctrl = BatchedLatTruck(cfg, batch_size=B)
    else:
        lat_ctrl.reset_state()
    if lon_ctrl is None:
        lon_ctrl = BatchedLonCtrl(cfg, batch_size=B)
    else:
        lon_ctrl.reset_state()

    # 挂车质量：优先用 override，否则读 cfg 默认
    tt_params = cfg['truck_trailer_vehicle']
    if trailer_mass_override is not None:
        if isinstance(trailer_mass_override, torch.Tensor):
            trailer_mass = trailer_mass_override.to(torch.float32).view(B)
        else:
            trailer_mass = torch.full((B,), float(trailer_mass_override))
    else:
        default_m = float(tt_params.get('default_trailer_mass_kg',
                                         tt_params.get('m_s_base', 0.0)))
        trailer_mass = torch.full((B,), default_m)

    vehicle = BatchedTruckTrailerVehicle(
        cfg, batch_size=B,
        init_x=bt.init_x, init_y=bt.init_y,
        init_yaw=bt.init_yaw, init_v=bt.init_v,
        dt=dt, trailer_mass_kg=trailer_mass)

    wheelbase = vehicle._L_t
    steer_ratio = vehicle._steer_ratio

    # 预分配 history：用 list-of-[B] 再 stack，autograd 图节点数最少
    h_x, h_y, h_yaw, h_v = [], [], [], []
    h_steer, h_steer_fb, h_steer_ff = [], [], []
    h_acc, h_torque = [], []
    h_lat_err, h_head_err = [], []
    h_ref_x, h_ref_y = [], []

    prev_steer = torch.zeros(B)
    v_prev = bt.init_v.clone()

    for step in range(T_max):
        t_now = torch.full((B,), step * dt)

        if tbptt_k > 0 and step > 0 and step % tbptt_k == 0:
            vehicle.detach_state()
            lat_ctrl.detach_state()
            lon_ctrl.detach_state()
            prev_steer = prev_steer.detach()
            v_prev = v_prev.detach()

        # yawrate 估计（用上一步 steer）
        delta_prev = prev_steer / steer_ratio * DEG2RAD
        yawrate = vehicle.v * torch.tan(delta_prev) / wheelbase

        steer_out, _kappa_cur, _nk, curvature_far, steer_fb, steer_ff = \
            lat_ctrl.compute(
                x=vehicle.x, y=vehicle.y,
                yaw_deg=vehicle.yaw_deg, speed_kph=vehicle.speed_kph,
                yawrate=yawrate, steer_feedback=prev_steer,
                btraj=bt, dt=dt, hard_mode=hard_mode)

        acc_cmd = lon_ctrl.compute(
            x=vehicle.x, y=vehicle.y,
            yaw_deg=vehicle.yaw_deg, speed_kph=vehicle.speed_kph,
            curvature_far=curvature_far,
            btraj=bt, t_now=t_now,
            ctrl_first_active=(step == 0), dt=dt, hard_mode=hard_mode)

        delta_front = steer_out / steer_ratio * DEG2RAD
        a_actual = (vehicle.v - v_prev) / dt
        torque_wheel = lon_ctrl.compute_torque_wheel(
            acc_cmd, vehicle.v, a_actual)

        # 当前步的 lateral / heading error（复用 lat_ctrl 内部的最近点计算）
        idx_curr = bt.query_nearest_idx(vehicle.x, vehicle.y)
        ref = bt.gather_ref(idx_curr)
        dx_e = vehicle.x - ref['x']
        dy_e = vehicle.y - ref['y']
        lateral_error = torch.cos(ref['theta']) * dy_e - torch.sin(ref['theta']) * dx_e
        heading_error = normalize_angle(vehicle.yaw - ref['theta'])

        h_x.append(vehicle.x)
        h_y.append(vehicle.y)
        h_yaw.append(vehicle.yaw)
        h_v.append(vehicle.v)
        h_steer.append(steer_out)
        h_steer_fb.append(steer_fb)
        h_steer_ff.append(steer_ff)
        h_acc.append(acc_cmd)
        h_torque.append(torque_wheel)
        h_lat_err.append(lateral_error)
        h_head_err.append(heading_error)
        h_ref_x.append(ref['x'])
        h_ref_y.append(ref['y'])

        v_prev = vehicle.v.detach()
        vehicle.step(delta=delta_front, torque_wheel=torque_wheel)
        prev_steer = steer_out

    # [T_max, B] → [B, T_max]
    def _stack(seq):
        return torch.stack(seq, dim=0).transpose(0, 1).contiguous()

    return {
        'x': _stack(h_x), 'y': _stack(h_y),
        'yaw': _stack(h_yaw), 'v': _stack(h_v),
        'steer': _stack(h_steer), 'steer_fb': _stack(h_steer_fb),
        'steer_ff': _stack(h_steer_ff),
        'acc': _stack(h_acc), 'torque_wheel': _stack(h_torque),
        'lateral_error': _stack(h_lat_err),
        'heading_error': _stack(h_head_err),
        'ref_x': _stack(h_ref_x), 'ref_y': _stack(h_ref_y),
        'valid_mask': bt.valid_mask,
        '_lat_ctrl': lat_ctrl, '_lon_ctrl': lon_ctrl, '_btraj': bt,
    }


def batched_tracking_loss(history: dict, ref_speeds: torch.Tensor,
                          w_lat=10.0, w_head=8.0, w_speed=3.0,
                          w_steer_rate=0.05, w_acc_rate=0.01,
                          return_details: bool = False):
    """Mask-aware 每轨迹跟踪 loss。返回 [B] per-traj loss。

    公式与 scalar tracking_loss 一致：各项误差在 valid 步上取 MSE，rate 项
    在相邻 valid 步对上取 MSE，加权相加。
    """
    mask = history['valid_mask']
    lat = history['lateral_error']
    head = history['heading_error']
    v = history['v']
    steer = history['steer']
    acc = history['acc']

    mask_sum = mask.sum(dim=1).clamp(min=1.0)

    lat_mse = (lat * lat * mask).sum(dim=1) / mask_sum
    head_mse = (head * head * mask).sum(dim=1) / mask_sum
    speed_err = v - ref_speeds[:, None]
    speed_mse = (speed_err * speed_err * mask).sum(dim=1) / mask_sum

    per_traj = w_lat * lat_mse + w_head * head_mse + w_speed * speed_mse

    dmask = mask[:, 1:] * mask[:, :-1]
    dmask_sum = dmask.sum(dim=1).clamp(min=1.0)
    dsteer = steer[:, 1:] - steer[:, :-1]
    dacc = acc[:, 1:] - acc[:, :-1]
    steer_rate_mse = (dsteer * dsteer * dmask).sum(dim=1) / dmask_sum
    acc_rate_mse = (dacc * dacc * dmask).sum(dim=1) / dmask_sum
    per_traj = (per_traj + w_steer_rate * steer_rate_mse
                + w_acc_rate * acc_rate_mse)

    if return_details:
        details = {
            'lat_rmse': lat_mse.sqrt().detach(),
            'head_rmse': head_mse.sqrt().detach(),
            'speed_rmse': speed_mse.sqrt().detach(),
            'lat_max': (lat.abs() * mask).max(dim=1).values.detach(),
            'head_max': (head.abs() * mask).max(dim=1).values.detach(),
            'loss_lat': (w_lat * lat_mse).detach(),
            'loss_head': (w_head * head_mse).detach(),
            'loss_speed': (w_speed * speed_mse).detach(),
            'loss_steer_rate': (w_steer_rate * steer_rate_mse).detach(),
            'loss_acc_rate': (w_acc_rate * acc_rate_mse).detach(),
        }
        return per_traj, details
    return per_traj


def _sanitize_grad_hook(grad):
    if grad is None:
        return None
    g = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return g.clamp(-1e4, 1e4)


def _export_tuned_config(lat_ctrl: BatchedLatTruck, lon_ctrl: BatchedLonCtrl,
                         cfg: dict) -> dict:
    """把批量控制器的 nn.Parameter 灌回 DiffControllerParams 再导出，
    保证 YAML 与 scalar 训练产物 byte-identical 兼容。"""
    p = DiffControllerParams(cfg=cfg)
    with torch.no_grad():
        for name in ['T2_y', 'T3_y', 'T4_y', 'T6_y']:
            getattr(p.lat_ctrl, name).copy_(getattr(lat_ctrl, name))
        for name in ['station_kp', 'station_ki', 'low_speed_kp',
                      'low_speed_ki', 'high_speed_kp', 'high_speed_ki',
                      'switch_speed']:
            getattr(p.lon_ctrl, name).copy_(getattr(lon_ctrl, name))
    return p.to_config_dict()


def _materialize_trajectories(trajectories):
    """接受 (a) None（默认全量 8×6=48）
             (b) list[str]（类型名，自动展开全速度段）
             (c) list[list[TrajectoryPoint]]（已生成的轨迹）。
    返回 [(key, traj)]。"""
    if trajectories is None:
        return [(key, gen()) for key, _lbl, gen in expand_trajectories(None)]
    if isinstance(trajectories, list) and trajectories:
        if isinstance(trajectories[0], str):
            return [(key, gen())
                    for key, _lbl, gen in expand_trajectories(trajectories)]
        if isinstance(trajectories[0], tuple):
            return [(k, t) for k, t in trajectories]
        # list[list[TrajectoryPoint]]
        return [(f'traj_{i}', t) for i, t in enumerate(trajectories)]
    raise ValueError(f"无法识别的 trajectories 参数：{type(trajectories)}")


def train_batch(trajectories=None, n_epochs: int = 100, lr: float = 5e-2,
                lr_tables: float = 5e-2, tbptt_k: int = 150,
                grad_clip: float = 10.0, verbose: bool = True,
                plant: str = None, config_path: str = None,
                w_lat: float = 10.0, w_head: float = 8.0, w_speed: float = 3.0,
                w_steer_rate: float = 0.05, w_acc_rate: float = 0.01,
                param_snapshot_interval: int = 10):
    """批量版训练入口。梯度/优化/归一化/投影/导出语义与 scalar train() 等价。

    返回 dict: {'losses', 'training_history', 'initial_params', 'final_params',
                'saved_path', 'trajectory_types', 'trajectory_keys',
                'lat_ctrl', 'lon_ctrl'}
    """
    cfg = load_config(config_path)
    if plant:
        apply_plant_override(cfg, plant)
    assert cfg['vehicle'].get('model_type') == 'truck_trailer', \
        "train_batch 目前仅支持 truck_trailer plant"

    pairs = _materialize_trajectories(trajectories)
    keys = [k for k, _t in pairs]
    trajs = [t for _k, t in pairs]
    B = len(trajs)

    if verbose:
        if isinstance(trajectories, list) and trajectories and isinstance(
                trajectories[0], str):
            n_types = len(trajectories)
        elif trajectories is None:
            n_types = len(TRAJECTORY_TYPES)
        else:
            n_types = B
        print(f"批量训练轨迹: {n_types} 类型 × {len(SPEED_BANDS_KPH)} 速度段 "
              f"= {B} 条 (batch 维 B={B})")
        print(f"  速度段: {SPEED_BANDS_KPH} kph")

    lat_ctrl = BatchedLatTruck(cfg, batch_size=B)
    lon_ctrl = BatchedLonCtrl(cfg, batch_size=B)

    ref_speeds = torch.tensor([float(t[0].v) for t in trajs])

    # 分组 lr（table y 用 lr_tables，其他用 lr）
    table_params, other_params = [], []
    all_named = list(lat_ctrl.named_parameters()) + list(lon_ctrl.named_parameters())
    for name, p in all_named:
        (table_params if '_y' in name else other_params).append(p)

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': table_params, 'lr': lr_tables},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.1)

    # 梯度清理钩子
    hooks = []
    for p in list(lat_ctrl.parameters()) + list(lon_ctrl.parameters()):
        hooks.append(p.register_hook(_sanitize_grad_hook))

    params_list = [p for _n, p in all_named]
    initial_params = {name: p.detach().clone() for name, p in all_named}

    losses = []
    training_history = []
    baseline_per_traj = None
    norm_floor = 1.0
    norm_alpha = 0.5
    t_start = time.time()

    for epoch in range(n_epochs):
        te = time.time()
        optimizer.zero_grad()

        history = run_simulation_batch(
            trajs, cfg=cfg, lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
            tbptt_k=tbptt_k)

        per_traj, details = batched_tracking_loss(
            history, ref_speeds,
            w_lat=w_lat, w_head=w_head, w_speed=w_speed,
            w_steer_rate=w_steer_rate, w_acc_rate=w_acc_rate,
            return_details=True)

        # Per-traj 软归一化（第 1 epoch 记 baseline，后续归一化）
        if epoch == 0:
            baseline_per_traj = per_traj.detach().clamp(min=1e-6)
            sorted_b = baseline_per_traj.sort().values
            median_b = sorted_b[len(sorted_b) // 2].item()
            norm_floor = median_b ** norm_alpha
            if verbose:
                print(f"  归一化下限: median_baseline={median_b:.4f}, "
                      f"norm_floor={norm_floor:.4f} (alpha={norm_alpha})")

        norm_factor = torch.maximum(
            baseline_per_traj ** norm_alpha,
            torch.tensor(norm_floor))
        weighted = (per_traj / norm_factor).mean()

        # L2 正则（对齐 scalar：(p - p_init)^2 总和 × 0.01）
        l2 = torch.zeros(())
        for name, p in all_named:
            l2 = l2 + ((p - initial_params[name]) ** 2).sum()
        total = weighted + 0.01 * l2
        total.backward()

        # NaN 梯度计数（钩子已 clamp，此处仅统计大值数）
        nan_count = 0
        for p in params_list:
            if p.grad is not None:
                nan_count += (p.grad.abs() >= 1e4).sum().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            params_list, max_norm=grad_clip).item()
        optimizer.step()

        # 参数投影
        with torch.no_grad():
            for name, p in lon_ctrl.named_parameters():
                if name in ('station_kp', 'station_ki', 'low_speed_kp',
                            'low_speed_ki', 'high_speed_kp', 'high_speed_ki'):
                    p.clamp_(min=0.0)
                elif name == 'switch_speed':
                    p.clamp_(min=0.5, max=10.0)
            for name, p in lat_ctrl.named_parameters():
                if name in ('T2_y', 'T3_y', 'T4_y', 'T6_y'):
                    p.clamp_(min=0.0)

        scheduler.step()

        loss_val = float(total.detach().item())
        losses.append(loss_val)
        dt_epoch = time.time() - te

        # 汇总轨迹明细（对齐 scalar 训练日志结构）
        per_trajectory = {}
        for i, key in enumerate(keys):
            per_trajectory[key] = {
                'lat_rmse': float(details['lat_rmse'][i]),
                'head_rmse': float(details['head_rmse'][i]),
                'speed_rmse': float(details['speed_rmse'][i]),
                'lat_max': float(details['lat_max'][i]),
                'head_max': float(details['head_max'][i]),
                'loss_lat': float(details['loss_lat'][i]),
                'loss_head': float(details['loss_head'][i]),
                'loss_speed': float(details['loss_speed'][i]),
                'loss_steer_rate': float(details['loss_steer_rate'][i]),
                'loss_acc_rate': float(details['loss_acc_rate'][i]),
            }
        avg = {k: float(details[k].mean().item()) for k in details}

        training_history.append({
            'epoch': epoch + 1,
            'loss': loss_val,
            'grad_norm': grad_norm,
            'nan_count': int(nan_count),
            'dt': dt_epoch,
            'per_trajectory': per_trajectory,
            'avg': avg,
        })

        if verbose:
            warn = f" [!NaN grads:{int(nan_count)}]" if nan_count > 0 else ""
            print(f"[{epoch+1:3d}/{n_epochs}] loss={loss_val:8.4f} | "
                  f"lat_rmse={avg['lat_rmse']:.4f}m "
                  f"head_rmse={avg['head_rmse']:.4f}rad "
                  f"spd_rmse={avg['speed_rmse']:.4f}m/s | "
                  f"grad_norm={grad_norm:.2f} "
                  f"dt={dt_epoch:.1f}s B={B}{warn}",
                  flush=True)

            if B > 1:
                for key in keys:
                    td = per_trajectory[key]
                    print(f"    {key:30s}: lat={td['lat_rmse']:.4f} "
                          f"head={td['head_rmse']:.4f} spd={td['speed_rmse']:.4f} | "
                          f"L_lat={td['loss_lat']:.3f} "
                          f"L_head={td['loss_head']:.3f} "
                          f"L_spd={td['loss_speed']:.3f}")

            if param_snapshot_interval > 0 and (epoch + 1) % param_snapshot_interval == 0:
                print(f"\n  --- 参数快照 (epoch {epoch+1}) ---")
                for name, p in all_named:
                    init_val = initial_params[name]
                    delta = p.detach() - init_val
                    if p.numel() == 1:
                        pct = (delta.item() / max(abs(init_val.item()), 1e-8)
                               * 100)
                        print(f"  {name:30s}: {init_val.item():.4f} -> "
                              f"{p.item():.4f} "
                              f"({delta.item():+.6f}, {pct:+.1f}%)")
                    else:
                        print(f"  {name:30s}: max_delta={delta.abs().max().item():.6f} "
                              f"mean={p.detach().mean().item():.4f} "
                              f"[{p.detach().min().item():.3f}, "
                              f"{p.detach().max().item():.3f}]")
                print()

    for h in hooks:
        h.remove()

    total_time = time.time() - t_start
    if verbose:
        print(f"\n训练完成! 总耗时: {total_time:.1f}s")
        if losses[0] > 0:
            print(f"  初始 loss: {losses[0]:.4f} → 最终 loss: {losses[-1]:.4f} "
                  f"(Δ={losses[-1]-losses[0]:+.4f}, "
                  f"{(losses[-1]-losses[0])/losses[0]*100:+.1f}%)")

    cfg_out = _export_tuned_config(lat_ctrl, lon_ctrl, cfg)
    if isinstance(trajectories, list) and trajectories and isinstance(
            trajectories[0], str):
        type_names = trajectories
    elif trajectories is None:
        type_names = list(TRAJECTORY_TYPES)
    else:
        type_names = keys

    saved_path = save_tuned_config(cfg_out, meta={
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'epochs': n_epochs,
        'trajectory_types': type_names,
        'trajectory_count': B,
        'speed_bands_kph': SPEED_BANDS_KPH,
        'lr': lr,
        'lr_tables': lr_tables,
        'tbptt_k': tbptt_k,
        'grad_clip': grad_clip,
        'w_lat': w_lat,
        'w_head': w_head,
        'w_speed': w_speed,
        'total_time_s': round(total_time, 1),
        'batched': True,
    })
    if verbose:
        print(f"参数已保存: {saved_path}")

    return {
        'losses': losses,
        'training_history': training_history,
        'trajectory_types': type_names,
        'trajectory_keys': keys,
        'initial_params': {name: p.cpu().tolist() if p.numel() > 1 else p.item()
                           for name, p in initial_params.items()},
        'final_params': {name: p.detach().cpu().tolist() if p.numel() > 1 else p.detach().item()
                         for name, p in all_named},
        'saved_path': saved_path,
        'lat_ctrl': lat_ctrl,
        'lon_ctrl': lon_ctrl,
    }


def _build_scalar_params_for_post_training(lat_ctrl, lon_ctrl, cfg):
    """post_training 需要 DiffControllerParams 风格的 params，这里灌一个。"""
    p = DiffControllerParams(cfg=cfg)
    with torch.no_grad():
        for name in ['T2_y', 'T3_y', 'T4_y', 'T6_y']:
            getattr(p.lat_ctrl, name).copy_(getattr(lat_ctrl, name))
        for name in ['station_kp', 'station_ki', 'low_speed_kp',
                      'low_speed_ki', 'high_speed_kp', 'high_speed_ki',
                      'switch_speed']:
            getattr(p.lon_ctrl, name).copy_(getattr(lon_ctrl, name))
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量可微调参训练 (truck_trailer)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--lr-tables', type=float, default=5e-2)
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='轨迹类型名（默认全量 8×6=48）')
    parser.add_argument('--tbptt-k', type=int, default=150)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--snapshot-interval', type=int, default=10)
    parser.add_argument('--plant', type=str, default=None,
                        choices=['truck_trailer'])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--w-lat', type=float, default=10.0)
    parser.add_argument('--w-head', type=float, default=8.0)
    parser.add_argument('--w-speed', type=float, default=3.0)
    parser.add_argument('--w-steer-rate', type=float, default=0.05)
    parser.add_argument('--w-acc-rate', type=float, default=0.01)
    parser.add_argument('--no-post-training', action='store_true',
                        help='训练完不跑 post_training 自动化')
    args = parser.parse_args()

    result = train_batch(
        trajectories=args.trajectories, n_epochs=args.epochs,
        lr=args.lr, lr_tables=args.lr_tables,
        tbptt_k=args.tbptt_k, grad_clip=args.grad_clip,
        plant=args.plant or 'truck_trailer', config_path=args.config,
        w_lat=args.w_lat, w_head=args.w_head, w_speed=args.w_speed,
        w_steer_rate=args.w_steer_rate, w_acc_rate=args.w_acc_rate,
        param_snapshot_interval=args.snapshot_interval)

    print(f"\n最终 loss: {result['losses'][-1]:.6f}")
    print(f"保存路径: {result['saved_path']}")

    if not args.no_post_training:
        # post_training 沿用 scalar 的 49 场景 V1 验证（truck_trailer plant）
        cfg = load_config(args.config)
        apply_plant_override(cfg, 'truck_trailer')
        scalar_params = _build_scalar_params_for_post_training(
            result['lat_ctrl'], result['lon_ctrl'], cfg)
        result['params'] = scalar_params  # post_training 约定字段
        from optim.post_training import run_post_training
        hyperparams = {
            'epochs': args.epochs, 'lr': args.lr, 'lr_tables': args.lr_tables,
            'trajectory_types': result['trajectory_types'],
            'tbptt_k': args.tbptt_k, 'grad_clip': args.grad_clip,
            'plant': 'truck_trailer',
            'w_lat': args.w_lat, 'w_head': args.w_head, 'w_speed': args.w_speed,
            'batched': True,
        }
        run_post_training(result, hyperparams, plant='truck_trailer',
                          trajectory_types=args.trajectories)
