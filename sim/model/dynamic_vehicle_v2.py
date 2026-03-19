# sim/model/dynamic_vehicle_v2.py
"""V2 动力学模型：逐轮胎力 + Euler 积分。

从 new_hybird_plant/ditui1.py 移植，适配 sim 仿真框架。
与 V1 (dynamic_vehicle.py) 的区别：
- forward() 返回 next_state（内含 Euler 积分 + yaw wrap），而非 derivatives
- 逐轮胎侧向力 + tanh 摩擦饱和（更高保真）
- 倒车方向因子（平滑 tanh 处理）
- 必须与 MLP checkpoint 训练时使用的 base model 完全一致

State: [x_f, y_f, yaw, vx_f, vy_f, r] — 前轴参考点
Control: [delta_sw, T_fl, T_fr, T_rl, T_rr] — 方向盘角 + 4 轮扭矩
"""
import torch
import torch.nn as nn


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """将角度限制在 [-pi, pi]。"""
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


class VehicleDynamicsV2(nn.Module):
    """V2 动力学模型：逐轮胎力 + Euler 积分。"""

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.m = float(params['mass'])
        self.Iz = float(params['Iz'])
        self.lf = float(params['lf'])
        self.lr = float(params['lr'])
        self.R = float(params['wheel_radius'])
        self.Cd = float(params['drag_coeff'])
        self.Af = float(params['frontal_area'])
        self.rho = float(params['air_density'])
        self.Crr = float(params['rolling_coeff'])
        self.C_alpha_f = float(params['corner_stiff_f'])
        self.C_alpha_r = float(params['corner_stiff_r'])
        self.mu = float(params.get('tire_friction_mu', 1.0))
        self.track_width = float(params['track_width'])
        self.steer_ratio = float(params.get('steer_ratio', 16.39))
        self.reverse_sign_speed_mps = max(
            float(params.get('reverse_sign_speed_mps', 0.5)), 1.0e-4)
        self.g = 9.81
        self.Fz_front = 0.5164 * self.m * self.g
        self.Fz_rear = 0.4836 * self.m * self.g
        self._speed_eps = 1.0e-8
        self._force_eps = 1.0e-8

    def forward(self, state: torch.Tensor, control: torch.Tensor,
                dt: torch.Tensor) -> torch.Tensor:
        """单步 Euler 积分。

        Args:
            state: [B, 6] — [x_f, y_f, yaw, vx_f, vy_f, r]
            control: [B, 5] — [delta_sw, T_fl, T_fr, T_rl, T_rr]
            dt: [B, 1] or scalar tensor

        Returns:
            next_state: [B, 6]
        """
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)

        # 解包
        x_f, y_f, yaw = state[:, 0], state[:, 1], state[:, 2]
        vx_f, vy_f, r = state[:, 3], state[:, 4], state[:, 5]

        # 方向盘角 → 前轮转角
        delta = control[:, 0] / self.steer_ratio
        torque_fl = control[:, 1]
        torque_fr = control[:, 2]
        torque_rl = control[:, 3]
        torque_rr = control[:, 4]

        # 前轴速度 → 后轴速度
        L = self.lf + self.lr
        vx_r = vx_f
        vy_r = vy_f - r * L

        # 后轴 → 质心速度
        vx_cg = vx_r
        vy_cg = vy_r + self.lr * r

        # 轮胎侧偏角 + 侧向力
        vx_mag = torch.sqrt(vx_r * vx_r + self.reverse_sign_speed_mps ** 2)
        alpha_f = torch.atan2(vy_cg + self.lf * r,
                              vx_mag + self._speed_eps) - delta
        alpha_r = torch.atan2(vy_cg - self.lr * r,
                              vx_mag + self._speed_eps)

        travel_dir = torch.tanh(vx_r / self.reverse_sign_speed_mps)

        fy_f0 = -self.C_alpha_f * alpha_f * travel_dir
        fy_r0 = -self.C_alpha_r * alpha_r * travel_dir

        # 摩擦饱和（tanh 平滑）
        fy_f_max = self.mu * (self.Fz_front * 0.5)
        fy_r_max = self.mu * (self.Fz_rear * 0.5)
        fy_f = fy_f_max * torch.tanh(fy_f0 / (fy_f_max + self._force_eps))
        fy_r = fy_r_max * torch.tanh(fy_r0 / (fy_r_max + self._force_eps))

        # 纵向力
        fx_fl = torque_fl / self.R
        fx_fr = torque_fr / self.R
        fx_rl = torque_rl / self.R
        fx_rr = torque_rr / self.R

        # 阻力
        speed = torch.sqrt(vx_cg * vx_cg + vy_cg * vy_cg + 1.0e-8)
        roll_force = self.Crr * self.m * self.g * torch.tanh(10.0 * vx_cg)
        aero_force = 0.5 * self.rho * self.Cd * self.Af * speed * speed

        # 前轮力旋转到车体系
        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)
        fx_front = (fx_fl + fx_fr) * cos_delta - (2.0 * fy_f) * sin_delta
        fy_front = (fx_fl + fx_fr) * sin_delta + (2.0 * fy_f) * cos_delta
        fx_rear = fx_rl + fx_rr
        fy_rear = 2.0 * fy_r

        fx_total = fx_front + fx_rear - roll_force - aero_force
        fy_total = fy_front + fy_rear

        yaw_moment = (
            fy_front * self.lf - fy_rear * self.lr
            + (fx_fr - fx_fl) * (self.track_width * 0.5)
            + (fx_rr - fx_rl) * (self.track_width * 0.5)
        )

        # 后轴参考点导数
        dvx_r = (fx_total + self.m * vy_r * r) / self.m
        dvy_r = (fy_total - self.m * vx_r * r) / self.m
        dr = yaw_moment / self.Iz

        # 后轴导数 → 前轴导数
        dvx_f = dvx_r
        dvy_f = dvy_r + dr * L

        # 运动学（前轴位置积分）
        dx_f = vx_f * torch.cos(yaw) - vy_f * torch.sin(yaw)
        dy_f = vx_f * torch.sin(yaw) + vy_f * torch.cos(yaw)
        dyaw = r

        derivatives = torch.stack(
            [dx_f, dy_f, dyaw, dvx_f, dvy_f, dr], dim=1)
        next_state = state + derivatives * dt
        next_state = next_state.clone()
        next_state[:, 2] = _wrap_angle(next_state[:, 2])
        return next_state
