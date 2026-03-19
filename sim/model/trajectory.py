# sim/model/trajectory.py
"""轨迹生成与分析器。V2: TrajectoryAnalyzer torch 化，生成函数不变。"""
import math
import torch
from common import TrajectoryPoint, normalize_angle, lookup1d


def generate_straight(length: float, speed: float, dt: float = 0.02,
                      heading: float = 0.0) -> list[TrajectoryPoint]:
    """生成直线轨迹。heading 为行驶方向（rad）。"""
    n_steps = int(length / (speed * dt))
    pts = []
    s = 0.0
    for i in range(n_steps + 1):
        t = i * dt
        x = speed * t * math.cos(heading)
        y = speed * t * math.sin(heading)
        pts.append(TrajectoryPoint(x=x, y=y, theta=heading, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
    return pts


def generate_circle(radius: float, speed: float,
                    arc_angle: float = 2 * math.pi,
                    dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成圆弧轨迹。从 (0,0) 开始，初始航向 0，圆心在 (0, R)。"""
    kappa = 1.0 / radius
    arc_length = radius * arc_angle
    n_steps = int(arc_length / (speed * dt))
    pts = []
    cx, cy = 0.0, radius
    s = 0.0
    for i in range(n_steps + 1):
        t = i * dt
        angle = speed * t / radius
        x = cx + radius * math.sin(angle)
        y = cy - radius * math.cos(angle)
        theta = angle
        pts.append(TrajectoryPoint(x=x, y=y, theta=theta, kappa=kappa,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
    return pts


def generate_sine(amplitude: float, wavelength: float, n_waves: float,
                  speed: float, dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成正弦曲线轨迹。y = A * sin(2*pi*x / lambda)。"""
    total_x = wavelength * n_waves
    n_fine = int(total_x / 0.01)
    xs, ys = [], []
    for i in range(n_fine + 1):
        xi = total_x * i / n_fine
        yi = amplitude * math.sin(2 * math.pi * xi / wavelength)
        xs.append(xi)
        ys.append(yi)

    arc_s = [0.0]
    for i in range(1, len(xs)):
        ds = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
        arc_s.append(arc_s[-1] + ds)
    total_arc = arc_s[-1]

    n_steps = int(total_arc / (speed * dt))
    pts = []
    fine_idx = 0
    for step in range(n_steps + 1):
        target_s = step * speed * dt
        while fine_idx < len(arc_s) - 2 and arc_s[fine_idx + 1] < target_s:
            fine_idx += 1
        if fine_idx >= len(arc_s) - 1:
            fine_idx = len(arc_s) - 2
        frac = 0.0
        ds_seg = arc_s[fine_idx + 1] - arc_s[fine_idx]
        if ds_seg > 1e-9:
            frac = (target_s - arc_s[fine_idx]) / ds_seg
        x = xs[fine_idx] + frac * (xs[fine_idx + 1] - xs[fine_idx])
        y = ys[fine_idx] + frac * (ys[fine_idx + 1] - ys[fine_idx])

        k = 2 * math.pi / wavelength
        dydx = amplitude * k * math.cos(k * x)
        d2ydx2 = -amplitude * k * k * math.sin(k * x)
        theta = math.atan2(dydx, 1.0)
        kappa = d2ydx2 / (1 + dydx ** 2) ** 1.5

        pts.append(TrajectoryPoint(x=x, y=y, theta=theta, kappa=kappa,
                                   v=speed, a=0.0, s=target_s, t=step * dt))
    return pts


def generate_combined(speed: float, dt: float = 0.02,
                      seg3_length: float = 30.0,
                      radius: float = 30.0,
                      lead_in: float = 30.0) -> list[TrajectoryPoint]:
    """生成组合轨迹：直线(lead_in) -> 左转圆弧(R, 90度) -> 直线(seg3_length)。"""
    pts = []
    s = 0.0
    t = 0.0

    seg1_len = lead_in
    n1 = int(seg1_len / (speed * dt))
    for i in range(n1):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    R = radius
    arc = math.pi / 2
    arc_len = R * arc
    n2 = int(arc_len / (speed * dt))
    cx = pts[-1].x
    cy = pts[-1].y + R
    start_angle = -math.pi / 2
    for i in range(1, n2 + 1):
        angle = start_angle + (speed * i * dt) / R
        x = cx + R * math.cos(angle)
        y = cy + R * math.sin(angle)
        theta = angle + math.pi / 2
        pts.append(TrajectoryPoint(x=x, y=y, theta=theta, kappa=1.0/R,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    seg3_len = seg3_length
    n3 = int(seg3_len / (speed * dt))
    last = pts[-1]
    for i in range(1, n3 + 1):
        x = last.x + speed * i * dt * math.cos(last.theta)
        y = last.y + speed * i * dt * math.sin(last.theta)
        pts.append(TrajectoryPoint(x=x, y=y, theta=last.theta, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    return pts


def generate_lane_change(lane_width: float, change_length: float,
                         speed: float, lead_in: float = 30.0,
                         lead_out: float = 30.0,
                         dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成换道轨迹：直线 → 余弦换道 → 直线。

    横向位移采用余弦曲线 y = (d/2)*(1 - cos(pi*x/L))，C2 连续。
    lane_width: 换道横向位移 (m)，正=向左
    change_length: 换道段纵向长度 (m)
    lead_in / lead_out: 换道前后直线长度 (m)
    """
    # ---- 精细采样换道段 ----
    n_fine = max(int(change_length / 0.01), 1000)
    xs_lc, ys_lc = [], []
    for i in range(n_fine + 1):
        xi = change_length * i / n_fine
        yi = (lane_width / 2) * (1 - math.cos(math.pi * xi / change_length))
        xs_lc.append(xi)
        ys_lc.append(yi)

    # ---- 弧长累计 ----
    arc_s_lc = [0.0]
    for i in range(1, len(xs_lc)):
        ds = math.hypot(xs_lc[i] - xs_lc[i-1], ys_lc[i] - ys_lc[i-1])
        arc_s_lc.append(arc_s_lc[-1] + ds)
    lc_arc = arc_s_lc[-1]

    # ---- 各段步数 ----
    n_lead_in = int(lead_in / (speed * dt))
    n_lc = int(lc_arc / (speed * dt))
    n_lead_out = int(lead_out / (speed * dt))

    pts: list[TrajectoryPoint] = []
    s = 0.0
    t = 0.0

    # 1) lead-in 直线
    for i in range(n_lead_in):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    x_offset = pts[-1].x + speed * dt if pts else 0.0

    # 2) 换道段（弧长等速重采样）
    fine_idx = 0
    k_coeff = math.pi / change_length
    for step in range(1, n_lc + 1):
        target_s = step * speed * dt
        while fine_idx < len(arc_s_lc) - 2 and arc_s_lc[fine_idx + 1] < target_s:
            fine_idx += 1
        if fine_idx >= len(arc_s_lc) - 1:
            fine_idx = len(arc_s_lc) - 2
        ds_seg = arc_s_lc[fine_idx + 1] - arc_s_lc[fine_idx]
        frac = (target_s - arc_s_lc[fine_idx]) / ds_seg if ds_seg > 1e-9 else 0.0
        lx = xs_lc[fine_idx] + frac * (xs_lc[fine_idx + 1] - xs_lc[fine_idx])
        ly = ys_lc[fine_idx] + frac * (ys_lc[fine_idx + 1] - ys_lc[fine_idx])

        dydx = (lane_width / 2) * k_coeff * math.sin(k_coeff * lx)
        d2ydx2 = (lane_width / 2) * k_coeff ** 2 * math.cos(k_coeff * lx)
        theta = math.atan2(dydx, 1.0)
        kappa = d2ydx2 / (1 + dydx ** 2) ** 1.5

        pts.append(TrajectoryPoint(x=x_offset + lx, y=ly, theta=theta,
                                   kappa=kappa, v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    # 3) lead-out 直线（在新车道内）
    last = pts[-1]
    for i in range(1, n_lead_out + 1):
        x = last.x + speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=lane_width, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    return pts


def _append_straight(pts: list, length: float, speed: float, dt: float,
                     x0: float, y0: float, theta: float,
                     s0: float, t0: float) -> tuple[float, float]:
    """内部辅助：从 (x0, y0, theta) 开始追加直线段，返回 (s_end, t_end)。"""
    n = int(length / (speed * dt))
    s, t = s0, t0
    for i in range(1, n + 1):
        x = x0 + speed * i * dt * math.cos(theta)
        y = y0 + speed * i * dt * math.sin(theta)
        pts.append(TrajectoryPoint(x=x, y=y, theta=theta, kappa=0.0,
                                   v=speed, a=0.0, s=s + speed * dt, t=t + dt))
        s += speed * dt
        t += dt
    return s, t


def _append_arc(pts: list, radius: float, arc_angle: float, direction: str,
                speed: float, dt: float,
                x0: float, y0: float, theta0: float,
                s0: float, t0: float) -> tuple[float, float, float]:
    """内部辅助：从 (x0, y0, theta0) 开始追加弧线段。

    direction: 'left' (CCW, kappa>0) or 'right' (CW, kappa<0)
    返回 (s_end, t_end, theta_end)。
    """
    sign = 1.0 if direction == 'left' else -1.0
    kappa = sign / radius
    arc_length = radius * arc_angle
    n = int(arc_length / (speed * dt))

    normal_angle = theta0 + sign * math.pi / 2
    cx = x0 + radius * math.cos(normal_angle)
    cy = y0 + radius * math.sin(normal_angle)
    start_angle = math.atan2(y0 - cy, x0 - cx)

    s, t = s0, t0
    for i in range(1, n + 1):
        d_angle = sign * speed * i * dt / radius
        angle = start_angle + d_angle
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        theta = theta0 + d_angle
        s += speed * dt
        t += dt
        pts.append(TrajectoryPoint(x=x, y=y, theta=theta, kappa=kappa,
                                   v=speed, a=0.0, s=s, t=t))

    theta_end = theta0 + sign * arc_angle
    return s, t, theta_end


def generate_double_lane_change(lane_width: float, change_length: float,
                                speed: float, hold_length: float = 20.0,
                                lead_in: float = 30.0, lead_out: float = 30.0,
                                dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成双换道轨迹：直线 → 左换道 → 短直线 → 右换道(返回) → 直线。

    复用余弦换道公式 y = (d/2)*(1-cos(pi*x/L))。终点 y ≈ 0。
    """
    def _cosine_lc_resample(d, L, x_offset, y_offset, speed, dt, pts, s, t):
        """生成单次余弦换道段并追加到 pts，返回 (s, t)。"""
        n_fine = max(int(L / 0.01), 1000)
        xs_lc, ys_lc = [], []
        for i in range(n_fine + 1):
            xi = L * i / n_fine
            yi = (d / 2) * (1 - math.cos(math.pi * xi / L))
            xs_lc.append(xi)
            ys_lc.append(yi)
        arc_s_lc = [0.0]
        for i in range(1, len(xs_lc)):
            ds = math.hypot(xs_lc[i] - xs_lc[i-1], ys_lc[i] - ys_lc[i-1])
            arc_s_lc.append(arc_s_lc[-1] + ds)
        lc_arc = arc_s_lc[-1]
        n_lc = int(lc_arc / (speed * dt))
        fine_idx = 0
        k_coeff = math.pi / L
        for step in range(1, n_lc + 1):
            target_s = step * speed * dt
            while fine_idx < len(arc_s_lc) - 2 and arc_s_lc[fine_idx + 1] < target_s:
                fine_idx += 1
            if fine_idx >= len(arc_s_lc) - 1:
                fine_idx = len(arc_s_lc) - 2
            ds_seg = arc_s_lc[fine_idx + 1] - arc_s_lc[fine_idx]
            frac = (target_s - arc_s_lc[fine_idx]) / ds_seg if ds_seg > 1e-9 else 0.0
            lx = xs_lc[fine_idx] + frac * (xs_lc[fine_idx + 1] - xs_lc[fine_idx])
            ly = ys_lc[fine_idx] + frac * (ys_lc[fine_idx + 1] - ys_lc[fine_idx])
            dydx = (d / 2) * k_coeff * math.sin(k_coeff * lx)
            d2ydx2 = (d / 2) * k_coeff ** 2 * math.cos(k_coeff * lx)
            theta = math.atan2(dydx, 1.0)
            kappa = d2ydx2 / (1 + dydx ** 2) ** 1.5
            s += speed * dt
            t += dt
            pts.append(TrajectoryPoint(x=x_offset + lx, y=y_offset + ly,
                                       theta=theta, kappa=kappa,
                                       v=speed, a=0.0, s=s, t=t))
        return s, t

    pts: list[TrajectoryPoint] = []
    s, t = 0.0, 0.0

    # 1) lead-in 直线
    n_in = int(lead_in / (speed * dt))
    for i in range(n_in):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    # 2) 第一次换道（左，y: 0 → lane_width）
    last = pts[-1]
    x_off = last.x + speed * dt
    s, t = _cosine_lc_resample(lane_width, change_length,
                               x_off, 0.0, speed, dt, pts, s, t)

    # 3) 保持直线（在新车道）
    last = pts[-1]
    n_hold = int(hold_length / (speed * dt))
    for i in range(1, n_hold + 1):
        x = last.x + speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=lane_width, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s + speed * dt, t=t + dt))
        s += speed * dt
        t += dt

    # 4) 第二次换道（右，y: lane_width → 0，d = -lane_width）
    last = pts[-1]
    x_off2 = last.x + speed * dt
    s, t = _cosine_lc_resample(-lane_width, change_length,
                               x_off2, lane_width, speed, dt, pts, s, t)

    # 5) lead-out 直线
    last = pts[-1]
    for i in range(1, int(lead_out / (speed * dt)) + 1):
        x = last.x + speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s + speed * dt, t=t + dt))
        s += speed * dt
        t += dt

    return pts


def generate_s_curve(radius: float, arc_angle: float, speed: float,
                     lead_in: float = 20.0, lead_out: float = 20.0,
                     dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成 S 弯轨迹：直线 → 左转弧 → 右转弧 → 直线。

    对称 S 弯：出口航向 ≈ 入口航向 (0)。
    前半段 kappa > 0（左转），后半段 kappa < 0（右转）。
    """
    pts: list[TrajectoryPoint] = []
    s, t = 0.0, 0.0

    # lead-in 直线
    n_in = int(lead_in / (speed * dt))
    for i in range(n_in):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    # 左转弧
    last = pts[-1]
    s, t, theta_mid = _append_arc(pts, radius, arc_angle, 'left', speed, dt,
                                   last.x, last.y, last.theta, s, t)

    # 右转弧（对称：同样的 arc_angle，出口航向回到 0）
    last = pts[-1]
    s, t, theta_end = _append_arc(pts, radius, arc_angle, 'right', speed, dt,
                                   last.x, last.y, last.theta, s, t)

    # lead-out 直线
    last = pts[-1]
    s, t = _append_straight(pts, lead_out, speed, dt,
                            last.x, last.y, theta_end, s, t)

    return pts


def generate_clothoid_turn(radius: float, turn_angle: float, speed: float,
                           clothoid_ratio: float = 0.3,
                           lead_in: float = 30.0, lead_out: float = 30.0,
                           dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成带缓和曲线（clothoid）过渡的弯道轨迹。

    直道 → clothoid(κ: 0→κ_max) → 圆弧(κ_max) → clothoid(κ_max→0) → 直道。
    clothoid 段曲率线性变化，避免曲率瞬变。

    Args:
        radius: 弯道半径 (m)，> 0
        turn_angle: 总航向变化 (rad)，正=左转(CCW+)，负=右转
        speed: 恒定参考速度 (m/s)
        clothoid_ratio: 两段 clothoid 合计占总角度变化的比例 (0~1)
        lead_in/lead_out: 弯道前后直线段长度 (m)
    """
    sign = 1.0 if turn_angle > 0 else -1.0
    abs_angle = abs(turn_angle)
    kappa_max = sign / radius

    # 角度分配：clothoid 占比 + 圆弧占比
    theta_cl_each = clothoid_ratio * abs_angle / 2
    theta_arc = (1 - clothoid_ratio) * abs_angle

    # 各段弧长
    L_cl = 2 * theta_cl_each * radius
    L_arc = theta_arc * radius
    total_length = lead_in + L_cl + L_arc + L_cl + lead_out

    # 段边界（沿弧长）
    s_cl1_start = lead_in
    s_cl1_end = lead_in + L_cl
    s_arc_end = s_cl1_end + L_arc
    s_cl2_end = s_arc_end + L_cl

    # 精细离散化 + 数值积分
    ds_fine = 0.01
    n_fine = int(total_length / ds_fine) + 1
    xs = [0.0]
    ys = [0.0]
    thetas = [0.0]
    kappas = [0.0]
    x, y, theta = 0.0, 0.0, 0.0

    for i in range(1, n_fine + 1):
        s = i * ds_fine
        if s <= s_cl1_start:
            kappa = 0.0
        elif s <= s_cl1_end:
            frac = (s - s_cl1_start) / L_cl if L_cl > 0 else 1.0
            kappa = kappa_max * frac
        elif s <= s_arc_end:
            kappa = kappa_max
        elif s <= s_cl2_end:
            frac = (s - s_arc_end) / L_cl if L_cl > 0 else 1.0
            kappa = kappa_max * (1 - frac)
        else:
            kappa = 0.0

        theta += kappa * ds_fine
        x += math.cos(theta) * ds_fine
        y += math.sin(theta) * ds_fine
        xs.append(x)
        ys.append(y)
        thetas.append(theta)
        kappas.append(kappa)

    # 等速重采样
    n_steps = int(total_length / (speed * dt))
    pts = []
    for step in range(n_steps + 1):
        target_s = step * speed * dt
        idx = min(int(target_s / ds_fine), len(xs) - 2)
        frac = (target_s - idx * ds_fine) / ds_fine
        px = xs[idx] + frac * (xs[idx + 1] - xs[idx])
        py = ys[idx] + frac * (ys[idx + 1] - ys[idx])
        ptheta = thetas[idx] + frac * (thetas[idx + 1] - thetas[idx])
        pkappa = kappas[idx] + frac * (kappas[idx + 1] - kappas[idx])
        pts.append(TrajectoryPoint(x=px, y=py, theta=ptheta, kappa=pkappa,
                                   v=speed, a=0.0, s=target_s, t=step * dt))
    return pts


def generate_uturn(radius: float, speed: float,
                   clothoid_ratio: float = 0.3,
                   lead_in: float = 20.0, lead_out: float = 20.0,
                   dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成掉头（U-turn）轨迹：180° clothoid 弯道。"""
    return generate_clothoid_turn(
        radius=radius, turn_angle=math.pi, speed=speed,
        clothoid_ratio=clothoid_ratio,
        lead_in=lead_in, lead_out=lead_out, dt=dt)


def generate_stop_and_go(cruise_speed: float, accel_rate: float = 0.5,
                         decel_rate: float = 0.5,
                         cruise_in: float = 50.0, cruise_out: float = 50.0,
                         stop_duration: float = 2.0,
                         dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成停靠起步轨迹：直线上 巡航→减速→停车→起步→巡航。"""
    pts = []
    x, s, t = 0.0, 0.0, 0.0

    n1 = int(cruise_in / (cruise_speed * dt))
    for i in range(n1 + 1):
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=cruise_speed, a=0.0, s=s, t=t))
        x += cruise_speed * dt
        s += cruise_speed * dt
        t += dt

    decel_time = cruise_speed / decel_rate
    n2 = int(decel_time / dt)
    for i in range(1, n2 + 1):
        v = max(0.0, cruise_speed - decel_rate * i * dt)
        v_mid = max(0.0, cruise_speed - decel_rate * (i - 0.5) * dt)
        a = -decel_rate if v > 0 else 0.0
        dx = v_mid * dt
        x += dx
        s += dx
        t += dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=v, a=a, s=s, t=t))

    n3 = int(stop_duration / dt)
    for i in range(1, n3 + 1):
        t += dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=0.0, a=0.0, s=s, t=t))

    accel_time = cruise_speed / accel_rate
    n4 = int(accel_time / dt)
    for i in range(1, n4 + 1):
        v = min(cruise_speed, accel_rate * i * dt)
        v_mid = min(cruise_speed, accel_rate * (i - 0.5) * dt)
        a = accel_rate if v < cruise_speed else 0.0
        dx = v_mid * dt
        x += dx
        s += dx
        t += dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=v, a=a, s=s, t=t))

    n5 = int(cruise_out / (cruise_speed * dt))
    for i in range(1, n5 + 1):
        x += cruise_speed * dt
        s += cruise_speed * dt
        t += dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=cruise_speed, a=0.0, s=s, t=t))

    return pts


def _chain_segments(segments: list[list[TrajectoryPoint]]) -> list[TrajectoryPoint]:
    """将多段轨迹首尾相连。每段的起点坐标系变换到前一段的终点。"""
    if not segments or not segments[0]:
        return []
    result = list(segments[0])
    for seg in segments[1:]:
        if not seg:
            continue
        prev = result[-1]
        cos_t = math.cos(prev.theta)
        sin_t = math.sin(prev.theta)
        s_off = prev.s
        t_off = prev.t
        for p in seg[1:]:
            rx = cos_t * p.x - sin_t * p.y
            ry = sin_t * p.x + cos_t * p.y
            result.append(TrajectoryPoint(
                x=prev.x + rx, y=prev.y + ry,
                theta=prev.theta + p.theta,
                kappa=p.kappa, v=p.v, a=p.a,
                s=s_off + p.s, t=t_off + p.t))
    return result


def generate_park_route(cruise_speed: float = 4.17, turn_speed: float = 2.78,
                        accel_rate: float = 0.5, stop_duration: float = 2.0,
                        dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成综合园区路线（含速度变化和启停）。

    路线：起步加速→直道巡航→减速右转→加速→直道→换道→直道→停靠→起步→
          直道→减速左转→加速→直道→减速停车。
    所有弯道带 clothoid 过渡，速度在弯前/弯后平滑过渡。
    """
    clothoid_ratio = 0.3
    R1, angle1 = 40.0, math.pi / 2
    R2, angle2 = 35.0, math.pi / 2

    def _turn_lengths(R, angle):
        theta_cl = clothoid_ratio * angle / 2
        L_cl = 2 * theta_cl * R
        L_arc = (1 - clothoid_ratio) * angle * R
        return L_cl, L_arc, 2 * L_cl + L_arc

    L_cl1, L_arc1, turn1_len = _turn_lengths(R1, angle1)
    L_cl2, L_arc2, turn2_len = _turn_lengths(R2, angle2)

    seg_defs = [
        ('S', 80.0),
        ('T', turn1_len, -1.0 / R1, L_cl1, L_arc1),
        ('S', 60.0),
        ('LC', 30.0, 3.5),
        ('S', 40.0),
        ('S', 40.0),
        ('T', turn2_len, 1.0 / R2, L_cl2, L_arc2),
        ('S', 60.0),
    ]

    seg_bounds = [0.0]
    for sd in seg_defs:
        seg_bounds.append(seg_bounds[-1] + sd[1])
    total_length = seg_bounds[-1]

    def kappa_at(s):
        for i, sd in enumerate(seg_defs):
            if s < seg_bounds[i] or s > seg_bounds[i + 1]:
                continue
            ls = s - seg_bounds[i]
            if sd[0] == 'S':
                return 0.0
            elif sd[0] == 'T':
                _, length, kmax, L_cl, L_arc = sd
                if ls <= L_cl:
                    return kmax * (ls / L_cl) if L_cl > 0 else kmax
                elif ls <= L_cl + L_arc:
                    return kmax
                elif ls <= 2 * L_cl + L_arc:
                    return kmax * (1 - (ls - L_cl - L_arc) / L_cl)
                return 0.0
            elif sd[0] == 'LC':
                _, length, d = sd
                k = math.pi / length
                dydx = (d / 2) * k * math.sin(k * ls)
                d2ydx2 = (d / 2) * k * k * math.cos(k * ls)
                return d2ydx2 / (1 + dydx ** 2) ** 1.5
        return 0.0

    ds = 0.01
    n_fine = int(total_length / ds) + 1
    fine_x = [0.0]
    fine_y = [0.0]
    fine_theta = [0.0]
    fine_kappa = [0.0]
    x, y, theta = 0.0, 0.0, 0.0
    for i in range(1, n_fine + 1):
        s_i = i * ds
        kappa = kappa_at(s_i)
        theta += kappa * ds
        x += math.cos(theta) * ds
        y += math.sin(theta) * ds
        fine_x.append(x)
        fine_y.append(y)
        fine_theta.append(theta)
        fine_kappa.append(kappa)

    def _accel_dist(v0, v1):
        return abs(v1 * v1 - v0 * v0) / (2 * accel_rate)

    d_start = _accel_dist(0, cruise_speed)
    d_to_turn = _accel_dist(turn_speed, cruise_speed)
    d_from_turn = d_to_turn
    d_stop = _accel_dist(cruise_speed, 0)

    turn1_s = seg_bounds[1]
    turn1_e = seg_bounds[2]
    stop_s = seg_bounds[5]
    turn2_s = seg_bounds[6]
    turn2_e = seg_bounds[7]

    speed_wps = [
        (0.0, 0.0),
        (d_start, cruise_speed),
        (turn1_s - d_to_turn, cruise_speed),
        (turn1_s, turn_speed),
        (turn1_e, turn_speed),
        (turn1_e + d_from_turn, cruise_speed),
        (stop_s - d_stop, cruise_speed),
        (stop_s, 0.0),
        (stop_s, 0.0),
        (stop_s + d_start, cruise_speed),
        (turn2_s - d_to_turn, cruise_speed),
        (turn2_s, turn_speed),
        (turn2_e, turn_speed),
        (turn2_e + d_from_turn, cruise_speed),
        (total_length - d_stop, cruise_speed),
        (total_length, 0.0),
    ]

    clean_wps = []
    for s_wp, v_wp in speed_wps:
        s_wp = max(0.0, min(s_wp, total_length))
        if clean_wps and s_wp < clean_wps[-1][0]:
            continue
        clean_wps.append((s_wp, v_wp))

    def speed_at(s):
        if s <= clean_wps[0][0]:
            return clean_wps[0][1]
        if s >= clean_wps[-1][0]:
            return clean_wps[-1][1]
        for j in range(len(clean_wps) - 1):
            s0, v0 = clean_wps[j]
            s1, v1 = clean_wps[j + 1]
            if s0 <= s <= s1:
                if s1 - s0 < 1e-6:
                    return v1
                frac = (s - s0) / (s1 - s0)
                return v0 + frac * (v1 - v0)
        return 0.0

    pts = []
    s_cur, t_cur = 0.0, 0.0
    v0 = speed_at(0.0)
    pts.append(TrajectoryPoint(x=0.0, y=0.0, theta=0.0,
                               kappa=kappa_at(0.0), v=v0, a=0.0, s=0.0, t=0.0))

    def _lookup_geom(s_val):
        idx = min(int(s_val / ds), len(fine_x) - 2)
        f = (s_val - idx * ds) / ds
        return (fine_x[idx] + f * (fine_x[idx + 1] - fine_x[idx]),
                fine_y[idx] + f * (fine_y[idx + 1] - fine_y[idx]),
                fine_theta[idx] + f * (fine_theta[idx + 1] - fine_theta[idx]),
                fine_kappa[idx] + f * (fine_kappa[idx + 1] - fine_kappa[idx]))

    def _append_stop(s_val, n_steps):
        gx, gy, gth, gk = _lookup_geom(s_val)
        for _ in range(n_steps):
            nonlocal t_cur
            t_cur += dt
            pts.append(TrajectoryPoint(x=gx, y=gy, theta=gth, kappa=gk,
                                       v=0.0, a=0.0, s=s_val, t=t_cur))

    V_SNAP = 0.15
    stop_done = False
    while s_cur < total_length - 1e-6:
        v = speed_at(s_cur)
        if not stop_done and v < V_SNAP and abs(s_cur - stop_s) < 2.0:
            s_cur = stop_s
            _append_stop(s_cur, int(stop_duration / dt))
            stop_done = True
            continue
        if v < V_SNAP and (total_length - s_cur) < 2.0:
            break
        ds_step = max(v, 0.05) * dt
        s_cur = min(s_cur + ds_step, total_length)
        t_cur += dt
        gx, gy, gth, gk = _lookup_geom(s_cur)
        v_now = speed_at(s_cur)
        a = (v_now - v) / dt if dt > 0 else 0.0
        pts.append(TrajectoryPoint(x=gx, y=gy, theta=gth, kappa=gk,
                                   v=v_now, a=a, s=s_cur, t=t_cur))

    return pts


def generate_offset_recovery(speed: float, lateral_offset: float = 1.5,
                              heading_error_deg: float = 5.0,
                              length: float = 200.0,
                              curvature: float = 0.0,
                              dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成偏移恢复轨迹：参考轨迹为直线或缓弯，车辆从偏移初始状态出发。

    测试控制器的阻尼/恢复能力。参考轨迹本身无偏移，初始状态偏移通过 sim_loop
    的 init_x/init_y/init_yaw 参数设置。此处返回的参考轨迹自身是直线或缓弯。

    Args:
        speed: 行驶速度 (m/s)
        lateral_offset: 横向偏移量 (m)，仅作为元数据记录
        heading_error_deg: 航向偏差 (deg)，仅作为元数据记录
        length: 轨迹总长 (m)
        curvature: 参考轨迹曲率 (1/m)，0 为直线
        dt: 时间步长 (s)
    """
    if abs(curvature) < 1e-6:
        # 直线参考轨迹
        return generate_straight(length=length, speed=speed, dt=dt)
    else:
        # 缓弯参考轨迹
        radius = 1.0 / abs(curvature)
        arc_angle = length / radius
        return generate_circle(radius=radius, speed=speed,
                               arc_angle=min(arc_angle, math.pi), dt=dt)


def generate_compound_curve(speed: float, radius: float = 50.0,
                            arc_angle: float = None,
                            straight_length: float = 30.0,
                            lead_in: float = 20.0, lead_out: float = 20.0,
                            dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成复合弯轨迹：直线 → 左转弧 → 直线 → 右转弧 → 直线。

    与 S 弯不同：中间有直线过渡段，测试曲率突变响应。
    """
    if arc_angle is None:
        arc_angle = math.pi / 4  # 45°

    pts: list[TrajectoryPoint] = []
    s, t = 0.0, 0.0

    # 1) lead-in 直线
    n_in = int(lead_in / (speed * dt))
    for i in range(n_in):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    # 2) 左转弧
    last = pts[-1]
    s, t, theta_mid1 = _append_arc(pts, radius, arc_angle, 'left', speed, dt,
                                    last.x, last.y, last.theta, s, t)

    # 3) 中间直线过渡
    last = pts[-1]
    s, t = _append_straight(pts, straight_length, speed, dt,
                            last.x, last.y, theta_mid1, s, t)

    # 4) 右转弧（同角度，出口航向回到约 0°）
    last = pts[-1]
    s, t, theta_mid2 = _append_arc(pts, radius, arc_angle, 'right', speed, dt,
                                    last.x, last.y, last.theta, s, t)

    # 5) lead-out 直线
    last = pts[-1]
    s, t = _append_straight(pts, lead_out, speed, dt,
                            last.x, last.y, theta_mid2, s, t)

    return pts


# ---------------------------------------------------------------------------
# 变速剖面后处理
# ---------------------------------------------------------------------------

def apply_curvature_speed_profile(
    pts: list[TrajectoryPoint],
    v_cruise: float,
    a_lat_max: float = 2.0,
    decel_rate: float = 0.8,
    accel_rate: float = 0.5,
) -> list[TrajectoryPoint]:
    """根据曲率约束生成弯前减速/弯后加速的速度剖面。

    算法：
      1. 曲率上限: v_max[i] = min(v_cruise, sqrt(a_lat_max / |κ|))
      2. 前向约束（减速）: v[i] <= sqrt(v[i-1]² + 2·decel_rate·ds)
      3. 后向约束（加速）: v[i] <= sqrt(v[i+1]² + 2·accel_rate·ds)
      4. 更新 v 和 a
    """
    n = len(pts)
    if n < 2:
        return pts

    eps = 1e-6
    # 1) 曲率速度上限
    v_max = [0.0] * n
    for i in range(n):
        kabs = abs(pts[i].kappa)
        v_curv = math.sqrt(a_lat_max / max(kabs, eps)) if kabs > eps else v_cruise
        v_max[i] = min(v_cruise, v_curv)

    # 2) 前向约束（减速限制）
    v_fwd = [0.0] * n
    v_fwd[0] = v_max[0]
    for i in range(1, n):
        ds = pts[i].s - pts[i - 1].s
        ds = max(ds, eps)
        v_lim = math.sqrt(v_fwd[i - 1] ** 2 + 2 * decel_rate * ds)
        v_fwd[i] = min(v_max[i], v_lim)

    # 3) 后向约束（加速限制）
    v_bwd = [0.0] * n
    v_bwd[-1] = v_fwd[-1]
    for i in range(n - 2, -1, -1):
        ds = pts[i + 1].s - pts[i].s
        ds = max(ds, eps)
        v_lim = math.sqrt(v_bwd[i + 1] ** 2 + 2 * accel_rate * ds)
        v_bwd[i] = min(v_fwd[i], v_lim)

    # 4) 生成新轨迹点
    dt = pts[1].t - pts[0].t if pts[1].t > pts[0].t else 0.02
    result = []
    for i in range(n):
        v_new = v_bwd[i]
        a_new = (v_bwd[i] - v_bwd[i - 1]) / dt if i > 0 and dt > 0 else 0.0
        result.append(TrajectoryPoint(
            x=pts[i].x, y=pts[i].y, theta=pts[i].theta, kappa=pts[i].kappa,
            v=v_new, a=a_new, s=pts[i].s, t=pts[i].t,
        ))
    return result


def apply_trapezoidal_speed_profile(
    pts: list[TrajectoryPoint],
    v_base: float,
    delta_ratio: float = 0.15,
    accel_rate: float = 0.5,
) -> list[TrajectoryPoint]:
    """在轨迹上叠加梯形速度波动：v_lo → 加速到 v_hi → 巡航 → 减速到 v_lo。

    速度剖面按弧长分配：
      [0, s1]: 起步巡航 v_lo
      [s1, s2]: 加速 v_lo → v_hi
      [s2, s3]: 高速巡航 v_hi
      [s3, s4]: 减速 v_hi → v_lo
      [s4, end]: 收尾巡航 v_lo
    """
    n = len(pts)
    if n < 2:
        return pts

    v_lo = v_base * (1.0 - delta_ratio)
    v_hi = v_base * (1.0 + delta_ratio)
    delta_v = v_hi - v_lo

    total_s = pts[-1].s - pts[0].s
    if total_s < 1e-3 or delta_v < 1e-6:
        return pts

    # 加减速所需距离: v_hi² - v_lo² = 2·a·d → d = (v_hi²-v_lo²)/(2·a)
    d_accel = (v_hi ** 2 - v_lo ** 2) / (2 * accel_rate)
    d_decel = d_accel  # 对称

    # 如果加减速距离超过总长的 80%，按比例缩小 delta_ratio
    if 2 * d_accel > 0.8 * total_s:
        d_accel = 0.4 * total_s
        d_decel = d_accel
        # 反算实际 v_hi: v_hi² = v_lo² + 2·a·d
        v_hi = math.sqrt(v_lo ** 2 + 2 * accel_rate * d_accel)
        delta_v = v_hi - v_lo

    # 弧长分段: 巡航(20%) → 加速 → 高速巡航 → 减速 → 巡航(20%)
    s0 = pts[0].s
    cruise_margin = 0.10 * total_s  # 前后各 10% 低速巡航
    s1 = s0 + cruise_margin                  # 加速起点
    s2 = s1 + d_accel                        # 加速终点
    s4 = s0 + total_s - cruise_margin        # 减速终点
    s3 = s4 - d_decel                        # 减速起点

    # 安全检查：确保 s2 <= s3
    if s2 > s3:
        s_mid = (s1 + s4) / 2
        s2 = s_mid
        s3 = s_mid

    dt = pts[1].t - pts[0].t if pts[1].t > pts[0].t else 0.02
    result = []
    for i in range(n):
        s = pts[i].s
        if s <= s1:
            v_new = v_lo
        elif s <= s2:
            frac = (s - s1) / max(s2 - s1, 1e-6)
            v_new = v_lo + frac * delta_v
        elif s <= s3:
            v_new = v_hi
        elif s <= s4:
            frac = (s - s3) / max(s4 - s3, 1e-6)
            v_new = v_hi - frac * delta_v
        else:
            v_new = v_lo

        a_new = (v_new - (result[-1].v if result else v_new)) / dt if dt > 0 else 0.0
        result.append(TrajectoryPoint(
            x=pts[i].x, y=pts[i].y, theta=pts[i].theta, kappa=pts[i].kappa,
            v=v_new, a=a_new, s=pts[i].s, t=pts[i].t,
        ))
    return result


# ---------------------------------------------------------------------------
# 速度段参数表 + 轨迹类型注册表
# ---------------------------------------------------------------------------

SPEED_BANDS_KPH = [5, 18, 25, 35, 45, 55]

# 每个速度段的几何参数
_SPEED_PARAMS = {
    5:  {'lc_len': 30, 'r_clothoid': 20, 'clothoid_angle': math.pi / 2,
         'r_combined': 15, 'r_scurve': 30, 'scurve_angle': math.pi / 4},
    18: {'lc_len': 50, 'r_clothoid': 40, 'clothoid_angle': math.pi / 2,
         'r_combined': 30, 'r_scurve': 50, 'scurve_angle': math.pi / 4},
    25: {'lc_len': 40, 'r_clothoid': 45, 'clothoid_angle': math.pi / 2,
         'r_combined': 35, 'r_scurve': 50, 'scurve_angle': math.pi / 4},
    35: {'lc_len': 55, 'r_clothoid': 50, 'clothoid_angle': math.pi / 2,
         'r_combined': 40, 'r_scurve': 60, 'scurve_angle': math.pi / 4},
    45: {'lc_len': 75, 'r_clothoid': 60, 'clothoid_angle': math.pi / 2,
         'r_combined': 50, 'r_scurve': 70, 'scurve_angle': math.pi / 4},
    55: {'lc_len': 90, 'r_clothoid': 70, 'clothoid_angle': math.pi / 3,
         'r_combined': 60, 'r_scurve': 80, 'scurve_angle': math.pi / 4},
}


def _build_trajectory(ttype: str, speed_kph: int) -> list[TrajectoryPoint]:
    """根据类型名和速度段生成轨迹。"""
    spd = speed_kph / 3.6
    p = _SPEED_PARAMS[speed_kph]

    if ttype == 'lane_change':
        return generate_lane_change(lane_width=3.5, change_length=p['lc_len'],
                                    speed=spd)
    elif ttype == 'double_lc':
        return generate_double_lane_change(lane_width=3.5,
                                           change_length=p['lc_len'], speed=spd)
    elif ttype == 'clothoid_left':
        return generate_clothoid_turn(radius=p['r_clothoid'],
                                      turn_angle=p['clothoid_angle'], speed=spd)
    elif ttype == 'clothoid_right':
        return generate_clothoid_turn(radius=p['r_clothoid'],
                                      turn_angle=-p['clothoid_angle'], speed=spd)
    elif ttype == 's_curve':
        return generate_s_curve(radius=p['r_scurve'],
                                arc_angle=p['scurve_angle'], speed=spd)
    elif ttype == 'combined_decel':
        # 使用速度段对应的弯道半径 + 充足的进入直线段（确保高速有足够减速距离）
        lead_in = max(30.0, spd * 6.0)  # 至少 6 秒预减速距离
        base = generate_combined(speed=spd, radius=p['r_combined'],
                                 lead_in=lead_in, seg3_length=lead_in)
        return apply_curvature_speed_profile(base, v_cruise=spd)
    elif ttype == 'clothoid_decel':
        # 用较小半径确保高速段有明显减速，加长 lead_in/out 给足加减速距离
        r_decel = max(15.0, p['r_clothoid'] * 0.6)
        lead = max(30.0, spd * 6.0)
        base = generate_clothoid_turn(radius=r_decel,
                                      turn_angle=p['clothoid_angle'], speed=spd,
                                      lead_in=lead, lead_out=lead)
        return apply_curvature_speed_profile(base, v_cruise=spd)
    elif ttype == 'lc_accel':
        base = generate_lane_change(lane_width=3.5, change_length=p['lc_len'],
                                    speed=spd)
        return apply_trapezoidal_speed_profile(base, v_base=spd)
    else:
        raise ValueError(f"Unknown trajectory type: {ttype}")


# 标准类型集（8 种）
TRAJECTORY_TYPES = [
    'lane_change', 'double_lc', 'clothoid_left', 'clothoid_right', 's_curve',
    'combined_decel', 'clothoid_decel', 'lc_accel',
]

# 中文标签（用于图表显示）
_TYPE_LABELS = {
    'lane_change': '单换道',
    'double_lc': '双换道',
    'clothoid_left': '左转clothoid',
    'clothoid_right': '右转clothoid',
    's_curve': 'S弯',
    'combined_decel': '组合弯(变速)',
    'clothoid_decel': 'clothoid(变速)',
    'lc_accel': '换道(变速)',
    'park_route': '园区综合',
}


def expand_trajectories(type_names: list[str] | None = None,
                        speed_bands: list[int] | None = None,
                        ) -> list[tuple[str, str, 'Callable']]:
    """将类型名展开为 (key, label, generator) 列表。

    Args:
        type_names: 轨迹类型列表，None 则使用全部 TRAJECTORY_TYPES
        speed_bands: 速度段列表 (kph)，None 则使用全部 SPEED_BANDS_KPH

    Returns:
        [(key, label, lazy_generator), ...] 其中 key 如 'lane_change_35kph',
        label 如 '单换道 (35kph)', lazy_generator 为无参函数返回轨迹点列表。
    """
    types = type_names if type_names is not None else TRAJECTORY_TYPES
    bands = speed_bands if speed_bands is not None else SPEED_BANDS_KPH

    result = []
    for tname in types:
        if tname not in _TYPE_LABELS:
            raise ValueError(
                f"Unknown trajectory type '{tname}'. "
                f"Available: {TRAJECTORY_TYPES}")
        for kph in bands:
            if kph not in _SPEED_PARAMS:
                raise ValueError(
                    f"Unknown speed band {kph} kph. "
                    f"Available: {SPEED_BANDS_KPH}")
            key = f"{tname}_{kph}kph"
            label = f"{_TYPE_LABELS[tname]} ({kph}kph)"
            # 用闭包捕获当前值
            gen = (lambda t=tname, k=kph: _build_trajectory(t, k))
            result.append((key, label, gen))
    return result


class TrajectoryAnalyzer:
    """轨迹分析器：位置查询、时间查询、Frenet 变换。V2: torch 化。"""

    def __init__(self, points: list[TrajectoryPoint]):
        self.points = points
        self._xs = torch.tensor([p.x for p in points])
        self._ys = torch.tensor([p.y for p in points])
        # 按时间索引的 tensor 数组（用于可微时间查询）
        self._t_arr = torch.tensor([p.t for p in points])
        self._kappa_arr = torch.tensor([p.kappa for p in points])
        self._v_arr = torch.tensor([p.v for p in points])
        self._a_arr = torch.tensor([p.a for p in points])
        self._s_arr = torch.tensor([p.s for p in points])

    def query_nearest_by_position(self, x, y) -> TrajectoryPoint:
        """最近点查询（detached argmin — 索引选择不在梯度路径上）。"""
        if isinstance(x, torch.Tensor):
            x_val = x.detach()
        else:
            x_val = torch.tensor(float(x))
        if isinstance(y, torch.Tensor):
            y_val = y.detach()
        else:
            y_val = torch.tensor(float(y))
        with torch.no_grad():
            dx = self._xs - x_val
            dy = self._ys - y_val
            dists = dx * dx + dy * dy
            idx = torch.argmin(dists).item()
        return self.points[idx]

    def query_nearest_by_relative_time(self, t_rel) -> TrajectoryPoint:
        """时间查询 — 逻辑与 V1 相同。输入可为 tensor 或 float。"""
        if isinstance(t_rel, torch.Tensor):
            t_rel = t_rel.item()
        if t_rel <= self.points[0].t:
            return self.points[0]
        if t_rel >= self.points[-1].t:
            return self.points[-1]
        for i in range(len(self.points) - 1):
            t0 = self.points[i].t
            t1 = self.points[i + 1].t
            if t0 <= t_rel <= t1:
                frac = (t_rel - t0) / (t1 - t0) if t1 > t0 else 0.0
                p0, p1 = self.points[i], self.points[i + 1]
                # normalize_angle 现在返回 tensor，取 .item() 用于 TrajectoryPoint
                theta_interp = p0.theta + frac * normalize_angle(
                    p1.theta - p0.theta).item()
                return TrajectoryPoint(
                    x=p0.x + frac * (p1.x - p0.x),
                    y=p0.y + frac * (p1.y - p0.y),
                    theta=theta_interp,
                    kappa=p0.kappa + frac * (p1.kappa - p0.kappa),
                    v=p0.v + frac * (p1.v - p0.v),
                    a=p0.a + frac * (p1.a - p0.a),
                    s=p0.s + frac * (p1.s - p0.s),
                    t=t_rel,
                )
        return self.points[-1]

    def query_by_time_differentiable(self, t: torch.Tensor):
        """可微时间查询 — 用 lookup1d 插值，返回 tensor，对 t 可微。
        用于 differentiable=True 路径，使 T5/T6 等时间参数可训练。
        返回 (kappa, v, a, s)，均为标量 tensor。
        """
        kappa = lookup1d(self._t_arr, self._kappa_arr, t)
        v = lookup1d(self._t_arr, self._v_arr, t)
        a = lookup1d(self._t_arr, self._a_arr, t)
        s = lookup1d(self._t_arr, self._s_arr, t)
        return kappa, v, a, s

    def to_frenet(self, x, y, theta_rad, v_mps, matched: TrajectoryPoint):
        """Frenet 变换 — torch 运算支持梯度流。
        返回 (s_matched, s_dot, d, d_dot) 全部为 torch.Tensor。
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(float(x))
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(float(y))
        if not isinstance(theta_rad, torch.Tensor):
            theta_rad = torch.tensor(float(theta_rad))
        if not isinstance(v_mps, torch.Tensor):
            v_mps = torch.tensor(float(v_mps))

        m_theta = torch.tensor(matched.theta)
        m_x = torch.tensor(matched.x)
        m_y = torch.tensor(matched.y)
        m_s = torch.tensor(matched.s)

        heading_err = normalize_angle(theta_rad - m_theta)
        s_dot = v_mps * torch.cos(heading_err)
        d = torch.cos(m_theta) * (y - m_y) - torch.sin(m_theta) * (x - m_x)
        d_dot = v_mps * torch.sin(heading_err)
        dx_v = x - m_x
        dy_v = y - m_y
        proj = dx_v * torch.cos(m_theta) + dy_v * torch.sin(m_theta)
        s_matched = m_s + proj
        return s_matched, s_dot, d, d_dot
