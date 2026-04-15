# sim/model/dynamic_vehicle.py
"""动力学车辆模型适配器。

将 6-DOF 动力学模型（VehicleDynamics）封装为与 BicycleModel 相同的接口：
- step(delta, acc): delta 是前轮转角(rad)，acc 是加速度(m/s²)
- 属性: x, y, yaw, v, speed_kph, yaw_deg
- detach_state(): 截断梯度链（Truncated BPTT）

内部将 acc 转换为后驱扭矩分配：T_rl = T_rr = (m * acc / 2) * R，T_fl = T_fr = 0。
使用 RK4 积分。
"""
import math

import torch
import torch.nn as nn


class VehicleDynamics(nn.Module):
    """6-DOF 动力学模型核心。

    state: [x_f, y_f, yaw, vx_f, vy_f, r] — 前轴参考点
    control: [delta_sw, T_fl, T_fr, T_rl, T_rr] — 方向盘角 + 四轮扭矩
    """
    def __init__(self, params):
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
        self.mu = float(params.get('tire_friction_mu', 0.85))
        self.track_width = float(params.get('track_width', 1.725))
        self.reverse_sign_speed_mps = max(
            float(params.get('reverse_sign_speed_mps', 0.5)), 1.0e-4)
        self.steer_ratio = max(
            float(params.get('steer_ratio', 16.39)), 1.0e-6)
        self.g = 9.81
        self.Fz_front = 0.5164 * self.m * self.g
        self.Fz_rear = 0.4836 * self.m * self.g
        self._speed_eps = 1.0e-8
        self._force_eps = 1.0e-8

    def derivatives(self, state, control):
        """计算状态导数。state: [B,6], control: [B,5] -> derivatives: [B,6]"""
        x_f, y_f, yaw = state[:, 0], state[:, 1], state[:, 2]
        vx_f, vy_f, r = state[:, 3], state[:, 4], state[:, 5]

        delta = control[:, 0] / self.steer_ratio
        torque_fl, torque_fr = control[:, 1], control[:, 2]
        torque_rl, torque_rr = control[:, 3], control[:, 4]

        L = self.lf + self.lr
        vx_r = vx_f
        vy_r = vy_f - r * L
        vx_cg = vx_r
        vy_cg = vy_r + self.lr * r

        vx_mag = torch.sqrt(vx_r * vx_r + self.reverse_sign_speed_mps ** 2)
        alpha_f = torch.atan2(vy_cg + self.lf * r, vx_mag + self._speed_eps) - delta
        alpha_r = torch.atan2(vy_cg - self.lr * r, vx_mag + self._speed_eps)

        travel_dir = torch.tanh(vx_r / self.reverse_sign_speed_mps)
        fy_f0 = -self.C_alpha_f * alpha_f * travel_dir
        fy_r0 = -self.C_alpha_r * alpha_r * travel_dir

        fy_f_max = self.mu * (self.Fz_front * 0.5)
        fy_r_max = self.mu * (self.Fz_rear * 0.5)
        fy_f = fy_f_max * torch.tanh(fy_f0 / (fy_f_max + self._force_eps))
        fy_r = fy_r_max * torch.tanh(fy_r0 / (fy_r_max + self._force_eps))

        fx_fl = torque_fl / self.R
        fx_fr = torque_fr / self.R
        fx_rl = torque_rl / self.R
        fx_rr = torque_rr / self.R

        speed = torch.sqrt(vx_cg * vx_cg + vy_cg * vy_cg + self._speed_eps)
        roll_force = self.Crr * self.m * self.g * torch.tanh(10.0 * vx_cg)
        aero_force = 0.5 * self.rho * self.Cd * self.Af * speed * speed

        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)

        fx_front = (fx_fl + fx_fr) * cos_delta - (2.0 * fy_f) * sin_delta
        fy_front = (fx_fl + fx_fr) * sin_delta + (2.0 * fy_f) * cos_delta
        fx_rear = fx_rl + fx_rr
        fy_rear = 2.0 * fy_r

        fx_total = fx_front + fx_rear - roll_force - aero_force
        fy_total = fy_front + fy_rear

        yaw_moment = (
            fy_front * self.lf
            - fy_rear * self.lr
            + (fx_fr - fx_fl) * (self.track_width * 0.5)
            + (fx_rr - fx_rl) * (self.track_width * 0.5)
        )

        dvx_r = (fx_total + self.m * vy_r * r) / self.m
        dvy_r = (fy_total - self.m * vx_r * r) / self.m
        dr = yaw_moment / self.Iz
        dvx_f = dvx_r
        dvy_f = dvy_r + dr * L

        dx_f = vx_f * torch.cos(yaw) - vy_f * torch.sin(yaw)
        dy_f = vx_f * torch.sin(yaw) + vy_f * torch.cos(yaw)
        dyaw = r

        return torch.stack([dx_f, dy_f, dyaw, dvx_f, dvy_f, dr], dim=1)

    def rk4_step(self, state, control, dt):
        """RK4 单步积分。"""
        k1 = self.derivatives(state, control)
        k2 = self.derivatives(state + 0.5 * dt * k1, control)
        k3 = self.derivatives(state + 0.5 * dt * k2, control)
        k4 = self.derivatives(state + dt * k3, control)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class DynamicVehicle:
    """动力学车辆适配器——与 BicycleModel 接口一致。

    内部维护 6D 状态 [x_f, y_f, yaw, vx, vy, r]（前轴参考点），
    对外暴露后轴坐标 x, y（通过 x_r = x_f - L*cos(yaw), y_r = y_f - L*sin(yaw) 转换），
    yaw 和 v（合速度）不变。
    """
    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0,
                 dt=0.02, differentiable=False):
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

    def step(self, delta, torque_wheel):
        """前进一步。

        Args:
            delta: 前轮转角 (rad)。适配器内部乘回 steer_ratio 得方向盘角。
            torque_wheel: 车轮总扭矩 (N·m)。由控制器的 compute_torque_wheel 输出，
                          后驱两轮平分。
        """
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(float(delta))
        if not isinstance(torque_wheel, torch.Tensor):
            torque_wheel = torch.tensor(float(torque_wheel))

        # delta(前轮转角) → 方向盘角
        delta_sw = delta * self._steer_ratio

        # 后驱两轮平分车轮总扭矩
        torque_rear = torque_wheel / 2.0
        zero = torch.zeros_like(torque_wheel)

        control = torch.stack([delta_sw, zero, zero, torque_rear, torque_rear])
        control = control.unsqueeze(0)  # [1, 5]

        state = self._state.unsqueeze(0)  # [1, 6]
        new_state = self.dynamics.rk4_step(state, control, self.dt)
        self._state = new_state.squeeze(0)

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
        """合速度 = sqrt(vx² + vy²)。"""
        vx = self._state[3]
        vy = self._state[4]
        return torch.sqrt(vx * vx + vy * vy + 1e-10)

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / 3.141592653589793)
