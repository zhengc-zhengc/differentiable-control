# sim/controller/lon.py
"""纵向控制器简化版 — 按 controller_spec.md §4.5 Steps 1-6 实现。
跳过 Steps 7-9（GearControl, CalFinalTorque）。
直接输出加速度指令 (m/s²)。参数从配置文件加载。
"""
import math
from common import lookup1d, clamp, normalize_angle, PID, IIR
from config import table_from_config
from trajectory import TrajectoryAnalyzer

DEG2RAD = math.pi / 180.0


class LonController:
    """纵向控制器（简化版：输出加速度，跳过 Gear/Torque）。"""

    def __init__(self, cfg: dict):
        lon = cfg['lon']

        self.station_kp = lon['station_kp']
        self.station_ki = lon['station_ki']
        self.low_speed_kp = lon['low_speed_kp']
        self.low_speed_ki = lon['low_speed_ki']
        self.high_speed_kp = lon['high_speed_kp']
        self.high_speed_ki = lon['high_speed_ki']
        self.switch_speed = lon['switch_speed']

        self.preview_window = lon['preview_window']
        self.preview_window_speed = lon['preview_window_speed']
        self.acc_use_preview_a = lon['acc_use_preview_a']
        self.station_error_limit = lon['station_error_limit']
        self.speed_input_limit = lon['speed_input_limit']
        self.acc_standstill_down_rate = lon['acc_standstill_down_rate']

        self.L1 = table_from_config(lon['L1_acc_up_lim'])
        self.L2 = table_from_config(lon['L2_acc_low_lim'])
        self.L3 = table_from_config(lon['L3_acc_up_rate'])
        self.L4 = table_from_config(lon['L4_acc_down_rate'])
        self.L5 = table_from_config(lon['L5_rate_gain'])

        pid_sat = lon['speed_pid_sat']
        self.station_pid = PID(
            kp=self.station_kp, ki=self.station_ki, kd=0.0,
            integrator_enable=lon['station_integrator_enable'],
            integrator_saturation=lon['station_sat'])
        self.speed_pid = PID(
            kp=self.low_speed_kp, ki=self.low_speed_ki, kd=0.0,
            integrator_enable=True, integrator_saturation=pid_sat)

        self.acc_out_prev = 0.0
        self.iir_acc = IIR(alpha=lon['iir_alpha'])
        self.station_error_fnl_prev = 0.0

    def compute(self, x: float, y: float, yaw_deg: float,
                speed_kph: float, accel_mps2: float,
                curvature_far: float,
                analyzer: TrajectoryAnalyzer, t_now: float,
                ctrl_enable: bool, ctrl_first_active: bool,
                dt: float = 0.02) -> float:
        """计算加速度指令 (m/s²)。"""
        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if ctrl_first_active:
            self.station_pid.reset()
            self.speed_pid.reset()
            self.iir_acc.reset()

        # Step 1: Frenet 误差
        matched = analyzer.query_nearest_by_position(x, y)
        s_match, s_dot, d, d_dot = analyzer.to_frenet(
            x, y, yaw_rad, speed_mps, matched)

        ref_pt = analyzer.query_nearest_by_relative_time(t_now)
        prev_pt = analyzer.query_nearest_by_relative_time(
            t_now + self.preview_window * dt)
        spd_pt = analyzer.query_nearest_by_relative_time(
            t_now + self.preview_window_speed * dt)

        station_error = ref_pt.s - s_match
        preview_speed_error = spd_pt.v - speed_mps
        preview_accel_ref = prev_pt.a

        # Step 2: 站位误差保护
        station_limited = clamp(station_error,
                                -self.station_error_limit,
                                self.station_error_limit)
        if speed_kph > 10:
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
        speed_offset = self.station_pid.control(station_fnl, dt)

        # Step 4: 速度 PID
        if speed_mps <= self.switch_speed:
            self.speed_pid.set_pid(self.low_speed_kp, self.low_speed_ki, 0.0)
        else:
            self.speed_pid.set_pid(self.high_speed_kp, self.high_speed_ki, 0.0)

        speed_input = clamp(speed_offset + preview_speed_error,
                            -self.speed_input_limit, self.speed_input_limit)
        acc_closeloop = self.speed_pid.control(speed_input, dt)

        # Step 5: 前馈叠加
        acc_cmd = acc_closeloop + self.acc_use_preview_a * preview_accel_ref

        # Step 6: CalFinalAccCmd
        if ctrl_enable:
            acc_up_lim = lookup1d(self.L1, abs(speed_kph))
            acc_low_lim = lookup1d(self.L2, abs(speed_kph))
            acc_up_rate_raw = lookup1d(self.L3, self.acc_out_prev)
            acc_dn_rate_raw = lookup1d(self.L4, self.acc_out_prev)
            rate_gain = lookup1d(self.L5, abs(speed_kph))
            acc_up_rate = acc_up_rate_raw * rate_gain

            if curvature_far < -0.0075:
                acc_up_lim *= 0.75
                acc_low_lim *= 0.60

            if abs(speed_mps) < 1.5:
                acc_dn_rate = self.acc_standstill_down_rate
            else:
                acc_dn_rate = acc_dn_rate_raw

            acc_clamped = clamp(acc_cmd, acc_low_lim, acc_up_lim)

            if abs(speed_mps) >= 0.2 or acc_clamped >= 0.25:
                acc_lowspd = acc_clamped
            else:
                acc_lowspd = min(-0.05, acc_clamped)

            acc_limited = clamp(acc_lowspd,
                                self.acc_out_prev + acc_dn_rate,
                                self.acc_out_prev + acc_up_rate)
            self.acc_out_prev = acc_limited
        else:
            acc_limited = 0.0
            self.acc_out_prev = 0.0

        acc_out = self.iir_acc.update(acc_limited)
        return acc_out
