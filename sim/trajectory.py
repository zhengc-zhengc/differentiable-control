# sim/trajectory.py
"""轨迹生成与分析器。"""
import math
from common import TrajectoryPoint, normalize_angle


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
    """生成组合轨迹：直线(30m) -> 左转圆弧(R=30m, 90°) -> 直线(seg3_length)。"""
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


class TrajectoryAnalyzer:
    """轨迹分析器：位置查询、时间查询、Frenet 变换。"""

    def __init__(self, points: list[TrajectoryPoint]):
        self.points = points

    def query_nearest_by_position(self, x: float, y: float) -> TrajectoryPoint:
        best_dist = float('inf')
        best_pt = self.points[0]
        for pt in self.points:
            dist = (pt.x - x) ** 2 + (pt.y - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pt = pt
        return best_pt

    def query_nearest_by_relative_time(self, t_rel: float) -> TrajectoryPoint:
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
                return TrajectoryPoint(
                    x=p0.x + frac * (p1.x - p0.x),
                    y=p0.y + frac * (p1.y - p0.y),
                    theta=p0.theta + frac * normalize_angle(p1.theta - p0.theta),
                    kappa=p0.kappa + frac * (p1.kappa - p0.kappa),
                    v=p0.v + frac * (p1.v - p0.v),
                    a=p0.a + frac * (p1.a - p0.a),
                    s=p0.s + frac * (p1.s - p0.s),
                    t=t_rel,
                )
        return self.points[-1]

    def to_frenet(self, x: float, y: float, theta_rad: float,
                  v_mps: float, matched: TrajectoryPoint
                  ) -> tuple[float, float, float, float]:
        heading_err = normalize_angle(theta_rad - matched.theta)
        s_dot = v_mps * math.cos(heading_err)
        d = (math.cos(matched.theta) * (y - matched.y)
             - math.sin(matched.theta) * (x - matched.x))
        d_dot = v_mps * math.sin(heading_err)
        dx_v = x - matched.x
        dy_v = y - matched.y
        proj = dx_v * math.cos(matched.theta) + dy_v * math.sin(matched.theta)
        s_matched = matched.s + proj
        return s_matched, s_dot, d, d_dot
