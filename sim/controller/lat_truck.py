# sim/controller/lat_truck.py
"""重卡横向控制器 — 按 controller_spec.md §2.5 实现。参数从配置文件加载。
V2: nn.Module 化，differentiable 开关控制梯度流。
"""
import torch
import torch.nn as nn
import math
from common import (lookup1d, rate_limit, clamp, sign, normalize_angle,
                    smooth_clamp, smooth_sign, smooth_lower_bound,
                    smooth_upper_bound, smooth_min)
from config import table_from_config
from model.trajectory import TrajectoryAnalyzer

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class LatControllerTruck(nn.Module):
    """重卡横向控制器。所有参数从 config dict 加载。
    可微参数（nn.Parameter）：T2-T6 查找表 y 值（控制设计参数）。
    固定参数（buffer）：kLh、T1/T7/T8 查找表 y 值（物理参数/安全约束）。
    differentiable=True 时全程保持 tensor 运算，梯度可流过 nn.Parameter。
    differentiable=False 时使用 .item() + Python if/else，与 V1 行为一致。
    """

    def __init__(self, cfg: dict, differentiable: bool = False):
        super().__init__()
        veh = cfg['vehicle']
        lat = cfg['lat_truck']

        self.wheelbase = veh['wheelbase']
        self.steer_ratio = veh['steer_ratio']
        self.differentiable = differentiable

        # 固定物理参数（不参与梯度优化）
        self.register_buffer('kLh', torch.tensor(float(lat['kLh'])))

        # 硬编码速率常量（不参与梯度）
        self.rate_limit_fb = lat['rate_limit_fb']
        self.rate_limit_ff = lat['rate_limit_ff']
        self.rate_limit_total = lat['rate_limit_total']
        self.min_prev_dist = lat['min_prev_dist']
        self.min_reach_dis = lat['min_reach_dis']
        self.min_speed_prot = lat['min_speed_prot']

        # 8 张查找表：x 为 buffer（固定断点）
        # T2-T6 的 y 值为 nn.Parameter（控制设计参数，参与梯度优化）
        # T1/T7/T8 的 y 值为 buffer（物理参数/安全约束，不参与优化）
        _fixed_tables = {'T1', 'T7', 'T8'}
        for name in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            key_map = {
                'T1': 'T1_max_theta_deg', 'T2': 'T2_prev_time_dist',
                'T3': 'T3_reach_time_theta', 'T4': 'T4_T_dt',
                'T5': 'T5_near_point_time', 'T6': 'T6_far_point_time',
                'T7': 'T7_max_steer_angle', 'T8': 'T8_slip_param',
            }
            xs, ys = table_from_config(lat[key_map[name]])
            self.register_buffer(f'{name}_x', xs)
            if name in _fixed_tables:
                self.register_buffer(f'{name}_y', ys)
            else:
                setattr(self, f'{name}_y', nn.Parameter(ys))

        # 内部状态（不参与梯度）
        self.register_buffer('steer_fb_prev', torch.tensor(0.0))
        self.register_buffer('steer_ff_prev', torch.tensor(0.0))
        self.register_buffer('steer_total_prev', torch.tensor(0.0))

    def reset_state(self):
        """重置控制器内部状态。"""
        self.steer_fb_prev.zero_()
        self.steer_ff_prev.zero_()
        self.steer_total_prev.zero_()

    def compute(self, x, y, yaw_deg, speed_kph, yawrate,
                steer_feedback, analyzer: TrajectoryAnalyzer,
                ctrl_enable: bool, dt: float = 0.02):
        """计算转向角。返回 (steering_target, kappa_current, kappa_near, kappa_far)。
        differentiable=True: 返回全 tensor。
        differentiable=False: 返回 float（V1 兼容）。
        """
        if self.differentiable:
            return self._compute_differentiable(
                x, y, yaw_deg, speed_kph, yawrate, steer_feedback,
                analyzer, ctrl_enable, dt)
        else:
            return self._compute_v1(
                x, y, yaw_deg, speed_kph, yawrate, steer_feedback,
                analyzer, ctrl_enable, dt)

    def _compute_v1(self, x, y, yaw_deg, speed_kph, yawrate,
                    steer_feedback, analyzer, ctrl_enable, dt):
        """V1 兼容路径：所有中间量为 float，与原始行为完全一致。"""
        # 确保是 float（从 tensor 或 float 输入均可）
        if isinstance(speed_kph, torch.Tensor):
            speed_kph_val = speed_kph.item()
        else:
            speed_kph_val = float(speed_kph)
        speed_kph_val = max(speed_kph_val, self.min_speed_prot)
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

        if isinstance(yawrate, torch.Tensor):
            yawrate_val = yawrate.item()
        else:
            yawrate_val = float(yawrate)

        if isinstance(steer_feedback, torch.Tensor):
            steer_feedback_val = steer_feedback.item()
        else:
            steer_feedback_val = float(steer_feedback)

        if not ctrl_enable:
            self.steer_fb_prev = torch.tensor(0.0)
            self.steer_ff_prev = torch.tensor(0.0)
            self.steer_total_prev = torch.tensor(float(steer_feedback_val))
            return steer_feedback_val, 0.0, 0.0, 0.0

        # Step 1: 查表 — lookup1d 接受 (x_tensor, y_tensor, x_scalar)
        speed_t = torch.tensor(speed_kph_val)
        max_theta_deg = lookup1d(self.T1_x, self.T1_y, speed_t).item()
        prev_time_dist = lookup1d(self.T2_x, self.T2_y, speed_t).item()
        reach_time_theta = lookup1d(self.T3_x, self.T3_y, speed_t).item()
        T_dt = lookup1d(self.T4_x, self.T4_y, speed_t).item()
        near_pt_time = lookup1d(self.T5_x, self.T5_y, speed_t).item()
        far_pt_time = lookup1d(self.T6_x, self.T6_y, speed_t).item()
        max_steer_angle = lookup1d(self.T7_x, self.T7_y, speed_t).item()
        slip_param = lookup1d(self.T8_x, self.T8_y, speed_t).item()

        # Step 2: 轨迹查询
        currt = analyzer.query_nearest_by_position(x_val, y_val)
        near = analyzer.query_nearest_by_relative_time(currt.t + near_pt_time)
        far = analyzer.query_nearest_by_relative_time(currt.t + far_pt_time)

        # Step 3: 误差计算
        dx = x_val - currt.x
        dy = y_val - currt.y
        lateral_error = math.cos(currt.theta) * dy - math.sin(currt.theta) * dx
        heading_error = normalize_angle(yaw_rad - currt.theta).item()
        curvature_far = far.kappa

        # Step 4: real_theta
        vehicle_speed_clamped = clamp(
            torch.tensor(speed_mps), 1.0, 100.0).item()
        real_theta = heading_error + math.atan(
            self.kLh.item() * yawrate_val / vehicle_speed_clamped)

        # Step 5: real_dt_theta
        real_dt_theta = yawrate_val - curvature_far * speed_mps

        # Step 6: target_theta
        prev_dist = max(speed_mps * prev_time_dist, self.min_prev_dist)
        dis2lane = -lateral_error
        error_angle_raw = math.atan(dis2lane / prev_dist)
        max_err_angle = min(max_theta_deg * DEG2RAD, abs(error_angle_raw))
        target_theta = sign(error_angle_raw).item() * max_err_angle

        target_dt_theta = (math.sin(real_theta) * speed_mps * prev_dist
                           / (prev_dist ** 2 + dis2lane ** 2) * -1.0)

        # Step 7: target_curvature
        denom = max(reach_time_theta * speed_mps, self.min_reach_dis)
        target_curvature = ((target_theta - real_theta)
                            + (target_dt_theta - real_dt_theta) * T_dt) / denom

        # Step 8: 反馈转向角
        steer_fb_raw = (math.atan(target_curvature * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_fb = rate_limit(self.steer_fb_prev, torch.tensor(steer_fb_raw),
                              self.rate_limit_fb, dt)
        self.steer_fb_prev = steer_fb.detach().clone()

        # Step 9: 前馈转向角
        steer_ff_raw = (math.atan(curvature_far * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_ff = rate_limit(self.steer_ff_prev, torch.tensor(steer_ff_raw),
                              self.rate_limit_ff, dt)
        self.steer_ff_prev = steer_ff.detach().clone()

        # Step 10: 合并输出
        steer_raw = clamp(steer_fb + steer_ff,
                          -max_steer_angle, max_steer_angle).item()
        steer_out = rate_limit(self.steer_total_prev, torch.tensor(steer_raw),
                               self.rate_limit_total, dt)
        self.steer_total_prev = steer_out.detach().clone()

        return steer_out.item(), currt.kappa, near.kappa, curvature_far

    def _compute_differentiable(self, x, y, yaw_deg, speed_kph, yawrate,
                                steer_feedback, analyzer, ctrl_enable, dt):
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
        if not isinstance(yawrate, torch.Tensor):
            yawrate = torch.tensor(float(yawrate))
        if not isinstance(steer_feedback, torch.Tensor):
            steer_feedback = torch.tensor(float(steer_feedback))

        # 速度保护：soft lower bound（避免 smooth_clamp 的宽区间梯度消失）
        speed_kph_safe = smooth_lower_bound(speed_kph, self.min_speed_prot)
        speed_mps = speed_kph_safe / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if not ctrl_enable:
            self.steer_fb_prev = torch.tensor(0.0)
            self.steer_ff_prev = torch.tensor(0.0)
            self.steer_total_prev = steer_feedback.detach().clone()
            zero = torch.tensor(0.0)
            return steer_feedback, zero, zero, zero

        # Step 1: 查表（结果为 tensor，保持梯度）
        max_theta_deg = lookup1d(self.T1_x, self.T1_y, speed_kph_safe)
        prev_time_dist = lookup1d(self.T2_x, self.T2_y, speed_kph_safe)
        reach_time_theta = lookup1d(self.T3_x, self.T3_y, speed_kph_safe)
        T_dt = lookup1d(self.T4_x, self.T4_y, speed_kph_safe)
        near_pt_time = lookup1d(self.T5_x, self.T5_y, speed_kph_safe)
        far_pt_time = lookup1d(self.T6_x, self.T6_y, speed_kph_safe)
        max_steer_angle = lookup1d(self.T7_x, self.T7_y, speed_kph_safe)
        slip_param = lookup1d(self.T8_x, self.T8_y, speed_kph_safe)

        # Step 2: 轨迹查询（argmin 不在梯度路径上，返回 TrajectoryPoint）
        currt = analyzer.query_nearest_by_position(x, y)
        near_pt_time_val = near_pt_time.item()
        far_pt_time_val = far_pt_time.item()
        near = analyzer.query_nearest_by_relative_time(
            currt.t + near_pt_time_val)
        far = analyzer.query_nearest_by_relative_time(
            currt.t + far_pt_time_val)

        # Step 3: 误差计算（tensor 运算）
        dx = x - currt.x
        dy = y - currt.y
        cos_theta = math.cos(currt.theta)
        sin_theta = math.sin(currt.theta)
        lateral_error = cos_theta * dy - sin_theta * dx
        heading_error = normalize_angle(yaw_rad - currt.theta)
        curvature_far_val = far.kappa
        curvature_far_t = torch.tensor(curvature_far_val)

        # Step 4: real_theta
        vehicle_speed_clamped = smooth_clamp(speed_mps, 1.0, 100.0, temp=1.0)
        real_theta = heading_error + torch.atan(
            self.kLh * yawrate / vehicle_speed_clamped)

        # Step 5: real_dt_theta
        real_dt_theta = yawrate - curvature_far_t * speed_mps

        # Step 6: target_theta
        # smooth lower bound（单侧约束，梯度不消失）
        prev_dist = smooth_lower_bound(speed_mps * prev_time_dist,
                                       self.min_prev_dist)
        dis2lane = -lateral_error
        error_angle_raw = torch.atan(dis2lane / prev_dist)
        max_err_angle_limit = max_theta_deg * DEG2RAD
        # smooth min(abs(error_angle_raw), max_err_angle_limit)
        abs_error = torch.abs(error_angle_raw)
        max_err_angle = smooth_min(abs_error, max_err_angle_limit)
        target_theta = smooth_sign(error_angle_raw, temp=0.5) * max_err_angle

        target_dt_theta = (torch.sin(real_theta) * speed_mps * prev_dist
                           / (prev_dist ** 2 + dis2lane ** 2) * -1.0)

        # Step 7: target_curvature
        # smooth lower bound（单侧约束）
        denom = smooth_lower_bound(reach_time_theta * speed_mps,
                                   self.min_reach_dis)
        target_curvature = ((target_theta - real_theta)
                            + (target_dt_theta - real_dt_theta) * T_dt) / denom

        # Step 8: 反馈转向角（rate_limit 使用 straight-through clamp）
        steer_fb_raw = (torch.atan(target_curvature * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_fb = rate_limit(self.steer_fb_prev, steer_fb_raw,
                              self.rate_limit_fb, dt, differentiable=True)
        self.steer_fb_prev = steer_fb.detach().clone()

        # Step 9: 前馈转向角
        steer_ff_raw = (torch.atan(curvature_far_t * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_ff = rate_limit(self.steer_ff_prev, steer_ff_raw,
                              self.rate_limit_ff, dt, differentiable=True)
        self.steer_ff_prev = steer_ff.detach().clone()

        # Step 10: 合并输出
        steer_raw = smooth_clamp(steer_fb + steer_ff,
                                 -max_steer_angle, max_steer_angle, temp=1.0)
        steer_out = rate_limit(self.steer_total_prev, steer_raw,
                               self.rate_limit_total, dt, differentiable=True)
        self.steer_total_prev = steer_out.detach().clone()

        return (steer_out, torch.tensor(currt.kappa),
                torch.tensor(near.kappa), curvature_far_t)
