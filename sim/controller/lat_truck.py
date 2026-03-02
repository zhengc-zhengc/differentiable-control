# sim/controller/lat_truck.py
"""重卡横向控制器 — 按 controller_spec.md §2.5 实现。参数从配置文件加载。"""
import math
from common import lookup1d, rate_limit, clamp, sign, normalize_angle
from config import table_from_config
from trajectory import TrajectoryAnalyzer

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class LatControllerTruck:
    """重卡横向控制器。所有参数从 config dict 加载。"""

    def __init__(self, cfg: dict):
        veh = cfg['vehicle']
        lat = cfg['lat_truck']

        self.wheelbase = veh['wheelbase']
        self.steer_ratio = veh['steer_ratio']

        self.kLh = lat['kLh']
        self.rate_limit_fb = lat['rate_limit_fb']
        self.rate_limit_ff = lat['rate_limit_ff']
        self.rate_limit_total = lat['rate_limit_total']
        self.min_prev_dist = lat['min_prev_dist']
        self.min_reach_dis = lat['min_reach_dis']
        self.min_speed_prot = lat['min_speed_prot']

        self.T1 = table_from_config(lat['T1_max_theta_deg'])
        self.T2 = table_from_config(lat['T2_prev_time_dist'])
        self.T3 = table_from_config(lat['T3_reach_time_theta'])
        self.T4 = table_from_config(lat['T4_T_dt'])
        self.T5 = table_from_config(lat['T5_near_point_time'])
        self.T6 = table_from_config(lat['T6_far_point_time'])
        self.T7 = table_from_config(lat['T7_max_steer_angle'])
        self.T8 = table_from_config(lat['T8_slip_param'])

        self.steer_fb_prev = 0.0
        self.steer_ff_prev = 0.0
        self.steer_total_prev = 0.0

    def compute(self, x: float, y: float, yaw_deg: float,
                speed_kph: float, yawrate: float, steer_feedback: float,
                analyzer: TrajectoryAnalyzer, ctrl_enable: bool,
                dt: float = 0.02) -> tuple[float, float, float, float]:
        """计算转向角。返回 (steering_target, kappa_current, kappa_near, kappa_far)。"""
        speed_kph = max(speed_kph, self.min_speed_prot)
        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if not ctrl_enable:
            self.steer_fb_prev = 0.0
            self.steer_ff_prev = 0.0
            self.steer_total_prev = steer_feedback
            return steer_feedback, 0.0, 0.0, 0.0

        # Step 1: 查表
        max_theta_deg = lookup1d(self.T1, speed_kph)
        prev_time_dist = lookup1d(self.T2, speed_kph)
        reach_time_theta = lookup1d(self.T3, speed_kph)
        T_dt = lookup1d(self.T4, speed_kph)
        near_pt_time = lookup1d(self.T5, speed_kph)
        far_pt_time = lookup1d(self.T6, speed_kph)
        max_steer_angle = lookup1d(self.T7, speed_kph)
        slip_param = lookup1d(self.T8, speed_kph)

        # Step 2: 轨迹查询
        currt = analyzer.query_nearest_by_position(x, y)
        near = analyzer.query_nearest_by_relative_time(currt.t + near_pt_time)
        far = analyzer.query_nearest_by_relative_time(currt.t + far_pt_time)

        # Step 3: 误差计算
        dx = x - currt.x
        dy = y - currt.y
        lateral_error = math.cos(currt.theta) * dy - math.sin(currt.theta) * dx
        heading_error = normalize_angle(yaw_rad - currt.theta)
        curvature_far = far.kappa

        # Step 4: real_theta
        # 注意：原版 C++ 使用 CW+ 航向角（顺时针为正），公式为 real_theta = -heading_error。
        # 我们的自行车模型使用 CCW+（逆时针为正），需要取反：real_theta = +heading_error。
        vehicle_speed_clamped = clamp(speed_mps, 1.0, 100.0)
        real_theta = heading_error + math.atan(self.kLh * yawrate / vehicle_speed_clamped)

        # Step 5: real_dt_theta（同理取反）
        real_dt_theta = yawrate - curvature_far * speed_mps

        # Step 6: target_theta
        prev_dist = max(speed_mps * prev_time_dist, self.min_prev_dist)
        dis2lane = -lateral_error
        error_angle_raw = math.atan(dis2lane / prev_dist)
        max_err_angle = min(max_theta_deg * DEG2RAD, abs(error_angle_raw))
        target_theta = sign(error_angle_raw) * max_err_angle

        target_dt_theta = (math.sin(real_theta) * speed_mps * prev_dist
                           / (prev_dist ** 2 + dis2lane ** 2) * -1.0)

        # Step 7: target_curvature（CCW+ 下去掉前导负号）
        denom = max(reach_time_theta * speed_mps, self.min_reach_dis)
        target_curvature = ((target_theta - real_theta)
                            + (target_dt_theta - real_dt_theta) * T_dt) / denom

        # Step 8: 反馈转向角
        steer_fb_raw = (math.atan(target_curvature * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_fb = rate_limit(self.steer_fb_prev, steer_fb_raw, self.rate_limit_fb, dt)
        self.steer_fb_prev = steer_fb

        # Step 9: 前馈转向角
        steer_ff_raw = (math.atan(curvature_far * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_ff = rate_limit(self.steer_ff_prev, steer_ff_raw, self.rate_limit_ff, dt)
        self.steer_ff_prev = steer_ff

        # Step 10: 合并输出
        steer_raw = clamp(steer_fb + steer_ff, -max_steer_angle, max_steer_angle)
        steer_out = rate_limit(self.steer_total_prev, steer_raw, self.rate_limit_total, dt)
        self.steer_total_prev = steer_out

        return steer_out, currt.kappa, near.kappa, curvature_far
