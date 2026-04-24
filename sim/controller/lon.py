# sim/controller/lon.py
"""纵向控制器简化版 — 按 controller_spec.md §4.5 Steps 1-6 实现。
跳过 Steps 7-9（GearControl, CalFinalTorque）。
直接输出加速度指令 (m/s²)。参数从配置文件加载。
V2: nn.Module 化，differentiable 开关控制梯度流。
"""
import math
import torch
import torch.nn as nn
from common import (lookup1d, clamp, normalize_angle, PID, IIR,
                    smooth_step, smooth_clamp, smooth_lower_bound,
                    smooth_upper_bound, _straight_through_clamp)
from config import table_from_config
from model.trajectory import TrajectoryAnalyzer

DEG2RAD = math.pi / 180.0


class LonController(nn.Module):
    """纵向控制器（简化版：输出加速度，跳过 Gear/Torque）。
    可微参数（nn.Parameter）：7 个 PID 标量（station/speed kp/ki + switch_speed）。
    固定参数（buffer）：L1-L5 查找表 y 值（加速度限幅/速率限制，物理约束）。
    differentiable=True 时全程保持 tensor 运算，梯度可流过 nn.Parameter。
    differentiable=False 时使用 .item() + Python if/else，与 V1 行为一致。
    """

    def __init__(self, cfg: dict, differentiable: bool = False):
        super().__init__()
        lon = cfg['lon']
        self.differentiable = differentiable

        # nn.Parameter: PID 增益 + 切换速度
        self.station_kp = nn.Parameter(torch.tensor(float(lon['station_kp'])))
        self.station_ki = nn.Parameter(torch.tensor(float(lon['station_ki'])))
        self.low_speed_kp = nn.Parameter(torch.tensor(float(lon['low_speed_kp'])))
        self.low_speed_ki = nn.Parameter(torch.tensor(float(lon['low_speed_ki'])))
        self.high_speed_kp = nn.Parameter(torch.tensor(float(lon['high_speed_kp'])))
        self.high_speed_ki = nn.Parameter(torch.tensor(float(lon['high_speed_ki'])))
        self.switch_speed = nn.Parameter(torch.tensor(float(lon['switch_speed'])))

        # 非梯度参数
        self.preview_window = lon['preview_window']
        self.preview_window_speed = lon['preview_window_speed']
        self.acc_use_preview_a = lon['acc_use_preview_a']
        self.station_error_limit = lon['station_error_limit']
        self.speed_input_limit = lon['speed_input_limit']
        self.acc_standstill_down_rate = lon['acc_standstill_down_rate']

        # 5 张查找表：x 为 buffer, y 为 buffer（物理限制/安全约束，不参与优化）
        for name, key in [('L1', 'L1_acc_up_lim'), ('L2', 'L2_acc_low_lim'),
                          ('L3', 'L3_acc_up_rate'), ('L4', 'L4_acc_down_rate'),
                          ('L5', 'L5_rate_gain')]:
            xs, ys = table_from_config(lon[key])
            self.register_buffer(f'{name}_x', xs)
            self.register_buffer(f'{name}_y', ys)

        # PID 配置
        pid_sat = lon['speed_pid_sat']
        self.station_integrator_enable = lon['station_integrator_enable']
        self.station_sat = lon['station_sat']
        self.speed_pid_sat = pid_sat

        self.station_pid = PID()
        self.speed_pid = PID()

        self.register_buffer('acc_out_prev', torch.tensor(0.0))
        self.iir_acc = IIR(alpha=lon['iir_alpha'])
        self.station_error_fnl_prev = 0.0

        # 扭矩输出层常数（对齐 C++ CalFinalTorque，跳过坡度和传动比）
        # 全部作为 buffer，不参与梯度
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
        """重置控制器内部状态。"""
        self.station_pid.reset()
        self.speed_pid.reset()
        self.iir_acc.reset()
        self.acc_out_prev.zero_()
        self.station_error_fnl_prev = 0.0

    def compute(self, x, y, yaw_deg, speed_kph, accel_mps2,
                curvature_far, analyzer: TrajectoryAnalyzer, t_now,
                ctrl_enable: bool, ctrl_first_active: bool,
                dt: float = 0.02):
        """计算加速度指令 (m/s²)。"""
        if self.differentiable:
            return self._compute_differentiable(
                x, y, yaw_deg, speed_kph, accel_mps2, curvature_far,
                analyzer, t_now, ctrl_enable, ctrl_first_active, dt)
        else:
            return self._compute_v1(
                x, y, yaw_deg, speed_kph, accel_mps2, curvature_far,
                analyzer, t_now, ctrl_enable, ctrl_first_active, dt)

    def _compute_v1(self, x, y, yaw_deg, speed_kph, accel_mps2,
                    curvature_far, analyzer, t_now,
                    ctrl_enable, ctrl_first_active, dt):
        """V1 兼容路径：所有中间量为 float。"""
        if isinstance(speed_kph, torch.Tensor):
            speed_kph_val = speed_kph.item()
        else:
            speed_kph_val = float(speed_kph)
        speed_mps = speed_kph_val / 3.6

        if isinstance(yaw_deg, torch.Tensor):
            yaw_rad = yaw_deg.item() * DEG2RAD
        else:
            yaw_rad = float(yaw_deg) * DEG2RAD

        if isinstance(x, torch.Tensor):
            x_val = x.item()
        else:
            x_val = float(x)
        if isinstance(y, torch.Tensor):
            y_val = y.item()
        else:
            y_val = float(y)

        if isinstance(t_now, torch.Tensor):
            t_now_val = t_now.item()
        else:
            t_now_val = float(t_now)

        if isinstance(curvature_far, torch.Tensor):
            curvature_far_val = curvature_far.item()
        else:
            curvature_far_val = float(curvature_far)

        if ctrl_first_active:
            self.station_pid.reset()
            self.speed_pid.reset()
            self.iir_acc.reset()

        # Step 1: Frenet 误差
        matched = analyzer.query_nearest_by_position(x_val, y_val)
        s_match, s_dot, d, d_dot = analyzer.to_frenet(
            x_val, y_val, yaw_rad, speed_mps, matched)
        # 转为 float
        s_match = s_match.item() if isinstance(s_match, torch.Tensor) else s_match

        ref_pt = analyzer.query_nearest_by_relative_time(t_now_val)
        prev_pt = analyzer.query_nearest_by_relative_time(
            t_now_val + self.preview_window * dt)
        spd_pt = analyzer.query_nearest_by_relative_time(
            t_now_val + self.preview_window_speed * dt)

        station_error = ref_pt.s - s_match
        preview_speed_error = spd_pt.v - speed_mps
        preview_accel_ref = prev_pt.a

        # Step 2: 站位误差保护
        station_limited = clamp(torch.tensor(station_error),
                                -self.station_error_limit,
                                self.station_error_limit).item()
        if speed_kph_val > 1.0:
            station_fnl = station_limited
        elif station_limited <= 0.25:
            station_fnl = min(0.0, station_limited)
        elif station_limited >= 0.8:
            station_fnl = station_limited
        elif self.station_error_fnl_prev <= 0.01:
            station_fnl = self.station_error_fnl_prev
        else:
            station_fnl = station_limited
        self.station_error_fnl_prev = station_fnl

        # Step 3: 站位 PID
        speed_offset = self.station_pid.control(
            station_fnl, dt,
            kp=self.station_kp.item(), ki=self.station_ki.item(), kd=0.0,
            integrator_enable=self.station_integrator_enable,
            sat=self.station_sat)
        speed_offset_val = speed_offset.item() if isinstance(speed_offset, torch.Tensor) else speed_offset

        # Step 4: 速度 PID
        if speed_mps <= self.switch_speed.item():
            kp = self.low_speed_kp.item()
            ki = self.low_speed_ki.item()
        else:
            kp = self.high_speed_kp.item()
            ki = self.high_speed_ki.item()

        speed_input = clamp(
            torch.tensor(speed_offset_val + preview_speed_error),
            -self.speed_input_limit, self.speed_input_limit).item()
        acc_closeloop = self.speed_pid.control(
            speed_input, dt,
            kp=kp, ki=ki, kd=0.0,
            integrator_enable=True, sat=self.speed_pid_sat)
        acc_closeloop_val = acc_closeloop.item() if isinstance(acc_closeloop, torch.Tensor) else acc_closeloop

        # Step 5: 前馈叠加
        acc_cmd = acc_closeloop_val + self.acc_use_preview_a * preview_accel_ref

        # Step 6: CalFinalAccCmd
        if ctrl_enable:
            speed_t = torch.tensor(abs(speed_kph_val))
            acc_up_lim = lookup1d(self.L1_x, self.L1_y, speed_t).item()
            acc_low_lim = lookup1d(self.L2_x, self.L2_y, speed_t).item()
            acc_up_rate_raw = lookup1d(self.L3_x, self.L3_y,
                                       torch.tensor(self.acc_out_prev.item())).item()
            acc_dn_rate_raw = lookup1d(self.L4_x, self.L4_y,
                                       torch.tensor(self.acc_out_prev.item())).item()
            rate_gain = lookup1d(self.L5_x, self.L5_y, speed_t).item()
            acc_up_rate = acc_up_rate_raw * rate_gain

            if curvature_far_val < -0.0075:
                acc_up_lim *= 0.75
                acc_low_lim *= 0.60

            if abs(speed_mps) < 1.5:
                acc_dn_rate = self.acc_standstill_down_rate
            else:
                acc_dn_rate = acc_dn_rate_raw

            acc_clamped = clamp(torch.tensor(acc_cmd),
                                acc_low_lim, acc_up_lim).item()

            if abs(speed_mps) >= 0.2 or acc_clamped >= 0.06:
                acc_lowspd = acc_clamped
            else:
                acc_lowspd = min(-0.05, acc_clamped)

            acc_limited = clamp(
                torch.tensor(acc_lowspd),
                self.acc_out_prev.item() + acc_dn_rate,
                self.acc_out_prev.item() + acc_up_rate).item()
            self.acc_out_prev.fill_(acc_limited)
        else:
            acc_limited = 0.0
            self.acc_out_prev.zero_()

        acc_out = self.iir_acc.update(acc_limited)
        return acc_out.item() if isinstance(acc_out, torch.Tensor) else acc_out

    def _compute_differentiable(self, x, y, yaw_deg, speed_kph, accel_mps2,
                                curvature_far, analyzer, t_now,
                                ctrl_enable, ctrl_first_active, dt):
        """可微路径：全程保持 tensor 运算，梯度可回传至所有 nn.Parameter。"""
        # 确保输入为 tensor
        if not isinstance(speed_kph, torch.Tensor):
            speed_kph = torch.tensor(float(speed_kph))
        if not isinstance(yaw_deg, torch.Tensor):
            yaw_deg = torch.tensor(float(yaw_deg))
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(float(x))
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(float(y))
        if not isinstance(t_now, torch.Tensor):
            t_now = torch.tensor(float(t_now))
        if not isinstance(curvature_far, torch.Tensor):
            curvature_far = torch.tensor(float(curvature_far))

        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if ctrl_first_active:
            self.station_pid.reset()
            self.speed_pid.reset()
            self.iir_acc.reset()

        # Step 1: Frenet 误差（需要 detach 的部分在 analyzer 内处理）
        t_now_val = t_now.item()
        matched = analyzer.query_nearest_by_position(x, y)
        s_match, s_dot, d_frenet, d_dot = analyzer.to_frenet(
            x, y, yaw_rad, speed_mps, matched)

        # 可微时间查询
        t_now_t = torch.tensor(t_now_val)
        _, _, _, ref_s = analyzer.query_by_time_differentiable(t_now_t)
        _, _, prev_a, _ = analyzer.query_by_time_differentiable(
            t_now_t + self.preview_window * dt)
        _, spd_v, _, _ = analyzer.query_by_time_differentiable(
            t_now_t + self.preview_window_speed * dt)

        station_error = ref_s - s_match
        preview_speed_error = spd_v - speed_mps
        preview_accel_ref = prev_a

        # Step 2: 站位误差保护
        # 使用 straight-through clamp（硬限幅但梯度穿透）
        station_limited = _straight_through_clamp(
            station_error, -self.station_error_limit, self.station_error_limit)

        # 简化：differentiable 模式用 smooth_step 混合高速/低速路径
        # 高速（>1.0kph，spec v2）直接用 station_limited；低速复杂逻辑简化为 station_limited
        # （低速站位保护逻辑不在梯度主要路径上，简化处理）
        speed_kph_val = speed_kph.item()
        if speed_kph_val > 1.0:
            station_fnl = station_limited
        elif station_limited.item() <= 0.25:
            station_fnl = smooth_upper_bound(station_limited, 0.0)
        elif station_limited.item() >= 0.8:
            station_fnl = station_limited
        elif self.station_error_fnl_prev <= 0.01:
            station_fnl = torch.tensor(self.station_error_fnl_prev)
        else:
            station_fnl = station_limited
        self.station_error_fnl_prev = station_fnl.item() if isinstance(
            station_fnl, torch.Tensor) else station_fnl

        # Step 3: 站位 PID（传入 nn.Parameter）
        speed_offset = self.station_pid.control(
            station_fnl, dt,
            kp=self.station_kp, ki=self.station_ki, kd=0.0,
            integrator_enable=self.station_integrator_enable,
            sat=self.station_sat, differentiable=True)

        # Step 4: 速度 PID（smooth_step 混合低速/高速增益）
        # w_low = 1 - sigmoid((speed - switch) / temp): 接近1表示低速
        w_low = 1.0 - smooth_step(speed_mps, self.switch_speed, temp=0.5)
        kp = w_low * self.low_speed_kp + (1.0 - w_low) * self.high_speed_kp
        ki = w_low * self.low_speed_ki + (1.0 - w_low) * self.high_speed_ki

        speed_input_raw = speed_offset + preview_speed_error
        speed_input = _straight_through_clamp(
            speed_input_raw, -self.speed_input_limit, self.speed_input_limit)
        acc_closeloop = self.speed_pid.control(
            speed_input, dt,
            kp=kp, ki=ki, kd=0.0,
            integrator_enable=True, sat=self.speed_pid_sat,
            differentiable=True)

        # Step 5: 前馈叠加
        acc_cmd = acc_closeloop + self.acc_use_preview_a * preview_accel_ref

        # Step 6: CalFinalAccCmd
        if ctrl_enable:
            abs_speed_kph = torch.abs(speed_kph)
            acc_up_lim = lookup1d(self.L1_x, self.L1_y, abs_speed_kph)
            acc_low_lim = lookup1d(self.L2_x, self.L2_y, abs_speed_kph)
            acc_prev_val = self.acc_out_prev.item()
            acc_up_rate_raw = lookup1d(self.L3_x, self.L3_y,
                                       torch.tensor(acc_prev_val))
            acc_dn_rate_raw = lookup1d(self.L4_x, self.L4_y,
                                       torch.tensor(acc_prev_val))
            rate_gain = lookup1d(self.L5_x, self.L5_y, abs_speed_kph)
            acc_up_rate = acc_up_rate_raw * rate_gain

            # 曲率条件：smooth_step 混合
            # w_curv = sigmoid((-0.0075 - curvature_far) / temp): 越大表示急弯
            w_curv = smooth_step(-curvature_far, 0.0075, temp=0.01)
            acc_up_lim_adj = acc_up_lim * (1.0 - 0.25 * w_curv)
            acc_low_lim_adj = acc_low_lim * (1.0 - 0.40 * w_curv)

            # 停车下坡率：smooth_step 混合
            abs_speed = torch.abs(speed_mps)
            w_standstill = 1.0 - smooth_step(abs_speed, 1.5, temp=0.3)
            acc_dn_rate = (w_standstill * self.acc_standstill_down_rate
                           + (1.0 - w_standstill) * acc_dn_rate_raw)

            # 幅值限制（straight-through）
            acc_clamped = _straight_through_clamp(
                acc_cmd, acc_low_lim_adj, acc_up_lim_adj)

            # 低速蠕行保护：smooth_step 混合
            # if abs(speed) >= 0.2 or acc >= 0.06: pass, else min(-0.05, acc)
            # 0.06 = kMinAccForStarting (spec v2)
            w_pass = smooth_step(abs_speed, 0.2, temp=0.05)
            w_acc_ok = smooth_step(acc_clamped, 0.06, temp=0.05)
            w_normal = 1.0 - (1.0 - w_pass) * (1.0 - w_acc_ok)
            acc_creep = smooth_upper_bound(acc_clamped, -0.05)
            acc_lowspd = w_normal * acc_clamped + (1.0 - w_normal) * acc_creep

            # 速率限制（straight-through）
            acc_limited = _straight_through_clamp(
                acc_lowspd,
                acc_prev_val + acc_dn_rate,
                acc_prev_val + acc_up_rate)
            self.acc_out_prev.fill_(acc_limited.detach().item())
        else:
            acc_limited = torch.tensor(0.0)
            self.acc_out_prev.zero_()

        acc_out = self.iir_acc.update(acc_limited)
        return acc_out

    def compute_torque_wheel(self, acc_cmd, speed_mps, a_actual):
        """加速度指令 → 车轮总扭矩 (N·m)。

        公式对齐 C++ lon_controller.cc CalFinalTorque（L749-811），
        跳过坡度补偿（F_slope=0）和传动比/效率（到车轮扭矩即止）。

        Args:
            acc_cmd: 控制器输出的加速度指令 (m/s²)
            speed_mps: 当前车速 (m/s)
            a_actual: 实际加速度 (m/s²)，由 sim_loop 差分得到

        Returns:
            T_wheel: 车轮总扭矩 (N·m)，后驱下游需要分半
        """
        if self.differentiable:
            return self._compute_torque_differentiable(
                acc_cmd, speed_mps, a_actual)
        else:
            return self._compute_torque_v1(acc_cmd, speed_mps, a_actual)

    def _compute_torque_v1(self, acc_cmd, speed_mps, a_actual):
        """V1 兼容路径：float 运算，对照 C++ 行为。"""
        if isinstance(acc_cmd, torch.Tensor):
            a_cmd = float(acc_cmd.item())
        else:
            a_cmd = float(acc_cmd)
        if isinstance(speed_mps, torch.Tensor):
            v = float(speed_mps.item())
        else:
            v = float(speed_mps)
        if isinstance(a_actual, torch.Tensor):
            a_act = float(a_actual.item())
        else:
            a_act = float(a_actual)

        m = self.torque_veh_mass.item()
        g = self.torque_gravity.item()
        r = self.torque_wheel_rolling_radius.item()
        deadzone = self.torque_accel_deadzone.item()

        F_air = (0.5 * self.torque_coef_cd.item()
                 * self.torque_air_density.item()
                 * self.torque_frontal_area.item() * v * v)
        F_rolling = self.torque_coef_rolling.item() * m * g
        F_inertia = self.torque_coef_delta.item() * m * a_cmd
        F_resist = F_air + F_rolling + F_inertia

        F_P = self.torque_accel_to_torque_kp.item() * (a_cmd - a_act)

        T_raw = (F_resist + F_P) * r

        if a_cmd > deadzone:
            return T_raw
        else:
            return 0.0

    def _compute_torque_differentiable(self, acc_cmd, speed_mps, a_actual):
        """可微路径：全程 tensor，mask.detach() 处理死区。"""
        if not isinstance(acc_cmd, torch.Tensor):
            acc_cmd = torch.tensor(float(acc_cmd))
        if not isinstance(speed_mps, torch.Tensor):
            speed_mps = torch.tensor(float(speed_mps))
        if not isinstance(a_actual, torch.Tensor):
            a_actual = torch.tensor(float(a_actual))

        m = self.torque_veh_mass
        g = self.torque_gravity
        r = self.torque_wheel_rolling_radius

        F_air = (0.5 * self.torque_coef_cd * self.torque_air_density
                 * self.torque_frontal_area * speed_mps * speed_mps)
        F_rolling = self.torque_coef_rolling * m * g
        F_inertia = self.torque_coef_delta * m * acc_cmd
        F_resist = F_air + F_rolling + F_inertia

        F_P = self.torque_accel_to_torque_kp * (acc_cmd - a_actual)

        T_raw = (F_resist + F_P) * r

        # 死区：硬 mask，detach 使门控不进入梯度图
        mask = (acc_cmd > self.torque_accel_deadzone).float().detach()
        return mask * T_raw
