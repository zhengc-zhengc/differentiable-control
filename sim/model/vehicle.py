# sim/model/vehicle.py
"""运动学自行车模型。V2: torch 化。"""
import torch


class BicycleModel:
    """运动学自行车模型。
    状态：[x, y, yaw, v] — 全部为 torch.Tensor
    输入：前轮转角 delta (rad) + 纵向加速度 acc (m/s²)
    """
    def __init__(self, wheelbase: float, x: float = 0.0, y: float = 0.0,
                 yaw: float = 0.0, v: float = 0.0, dt: float = 0.02,
                 differentiable: bool = False):
        self.L = wheelbase
        self.x = torch.tensor(float(x))
        self.y = torch.tensor(float(y))
        self.yaw = torch.tensor(float(yaw))
        self.v = torch.tensor(float(v))
        self.dt = dt
        self.differentiable = differentiable

    def step(self, delta, acc):
        """前进一步。delta: 前轮转角(rad), acc: 加速度(m/s²)。"""
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(float(delta))
        if not isinstance(acc, torch.Tensor):
            acc = torch.tensor(float(acc))
        self.x = self.x + self.v * torch.cos(self.yaw) * self.dt
        self.y = self.y + self.v * torch.sin(self.yaw) * self.dt
        self.yaw = self.yaw + self.v * torch.tan(delta) / self.L * self.dt
        self.v = self.v + acc * self.dt
        if self.differentiable:
            self.v = torch.nn.functional.softplus(self.v, beta=10.0)
        else:
            self.v = torch.clamp(self.v, min=0.0)

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / 3.141592653589793)
