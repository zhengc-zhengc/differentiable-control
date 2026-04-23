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
        # 上一步末的横摆角速度（kinematic 模型没有 r 状态，在 step 里显式记录）
        self._yawrate = torch.tensor(0.0)
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
        # 步末 yawrate = v_new · tan(δ_applied) / L（与 sim_loop 原 synth 公式等价）
        self._yawrate = self.v * torch.tan(delta) / self.L

    def detach_state(self):
        """截断梯度链：将当前状态从计算图中 detach（用于 truncated BPTT）。"""
        self.x = self.x.detach().requires_grad_(False)
        self.y = self.y.detach().requires_grad_(False)
        self.yaw = self.yaw.detach().requires_grad_(False)
        self.v = self.v.detach().requires_grad_(False)
        self._yawrate = self._yawrate.detach().requires_grad_(False)

    @property
    def yawrate(self):
        """kinematic 模型的当前横摆角速度 = v · tan(δ_last) / L。"""
        return self._yawrate

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / 3.141592653589793)
