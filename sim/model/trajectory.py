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
                      seg3_length: float = 30.0) -> list[TrajectoryPoint]:
    """生成组合轨迹：直线(30m) -> 左转圆弧(R=30m, 90度) -> 直线(seg3_length)。"""
    pts = []
    s = 0.0
    t = 0.0

    seg1_len = 30.0
    n1 = int(seg1_len / (speed * dt))
    for i in range(n1):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    R = 30.0
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
