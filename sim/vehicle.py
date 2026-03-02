# sim/vehicle.py
"""运动学自行车模型。"""
import math


class BicycleModel:
    """运动学自行车模型。
    状态：[x, y, yaw, v]
    输入：前轮转角 delta (rad) + 纵向加速度 acc (m/s²)
    """
    def __init__(self, wheelbase: float, x: float = 0.0, y: float = 0.0,
                 yaw: float = 0.0, v: float = 0.0, dt: float = 0.02):
        self.L = wheelbase
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.dt = dt

    def step(self, delta: float, acc: float):
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v * math.tan(delta) / self.L * self.dt
        self.v += acc * self.dt
        self.v = max(0.0, self.v)

    @property
    def speed_kph(self) -> float:
        return self.v * 3.6

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)
