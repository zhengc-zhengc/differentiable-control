# sim/common.py
"""公共基础组件 — 按 controller_spec.md §1 实现。"""
import math
from dataclasses import dataclass


def lookup1d(table: list[tuple[float, float]], x: float) -> float:
    """分段线性插值，边界 clamp。table 为 (index, value) 有序列表。"""
    if len(table) == 1 or x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def rate_limit(prev: float, target: float, rate: float, dt: float) -> float:
    """速率限制器。rate 单位与 target/dt 相同。"""
    max_delta = rate * dt
    return prev + max(-max_delta, min(max_delta, target - prev))


def normalize_angle(angle: float) -> float:
    """归一化角度到 (-pi, pi]。"""
    a = math.fmod(angle, 2 * math.pi)
    if a > math.pi:
        a -= 2 * math.pi
    elif a <= -math.pi:
        a += 2 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sign(x: float) -> float:
    if abs(x) < 1e-3:
        return 0.0
    return 1.0 if x > 0 else -1.0


class PID:
    """PID 控制器，含抗积分饱和。按 spec §1.5。"""
    def __init__(self, kp: float, ki: float, kd: float,
                 integrator_enable: bool, integrator_saturation: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_enable = integrator_enable
        self.sat = integrator_saturation
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error: float, dt: float) -> float:
        if self.integrator_enable:
            self.integral = clamp(self.integral + error * dt, -self.sat, self.sat)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def set_pid(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd


class IIR:
    """一阶 IIR 低通滤波器（传递函数形式）。y = x - alpha * y_prev"""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y_prev = 0.0

    def update(self, x: float) -> float:
        y = x - self.alpha * self.y_prev
        self.y_prev = y
        return y

    def reset(self):
        self.y_prev = 0.0


@dataclass
class TrajectoryPoint:
    """轨迹点。"""
    x: float
    y: float
    theta: float
    kappa: float
    v: float
    a: float
    s: float
    t: float
