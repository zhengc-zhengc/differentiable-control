# sim/common.py
"""公共基础组件 — 按 controller_spec.md §1 实现。V2: torch 化。"""
import torch
from dataclasses import dataclass


def lookup1d(table_x: torch.Tensor, table_y: torch.Tensor,
             x: torch.Tensor) -> torch.Tensor:
    """分段线性插值，边界 clamp。
    table_x = 断点张量, table_y = 值张量（可为 nn.Parameter）。
    x 为标量张量。
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    if len(table_x) == 1:
        return table_y[0].clone()
    x_clamped = torch.clamp(x, table_x[0].item(), table_x[-1].item())
    idx = torch.searchsorted(table_x, x_clamped) - 1
    idx = torch.clamp(idx, 0, len(table_x) - 2)
    i = idx.long()
    x0, x1 = table_x[i], table_x[i + 1]
    y0, y1 = table_y[i], table_y[i + 1]
    t = (x_clamped - x0) / (x1 - x0 + 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    return y0 + (y1 - y0) * t


# ── 平滑函数（可微模式） ──────────────────────────────

def smooth_clamp(x: torch.Tensor, lo, hi, temp: float = 0.1) -> torch.Tensor:
    """tanh 平滑 clamp，边界处梯度非零。"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    lo = float(lo) if not isinstance(lo, torch.Tensor) else lo
    hi = float(hi) if not isinstance(hi, torch.Tensor) else hi
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if isinstance(half, torch.Tensor):
        half_val = half.item()
    else:
        half_val = half
    if half_val < 1e-12:
        return torch.tensor(float(mid))
    return mid + half * torch.tanh((x - mid) / (half * temp))


def smooth_lower_bound(x: torch.Tensor, lo, sharpness: float = 10.0) -> torch.Tensor:
    """softplus 平滑下界：soft_max(x, lo)。梯度不消失。
    sharpness 越大越接近 max(x, lo)。
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    lo_val = float(lo) if not isinstance(lo, torch.Tensor) else lo
    return lo_val + torch.nn.functional.softplus(
        (x - lo_val) * sharpness, beta=1.0) / sharpness


def smooth_upper_bound(x: torch.Tensor, hi, sharpness: float = 10.0) -> torch.Tensor:
    """softplus 平滑上界：soft_min(x, hi)。梯度不消失。
    sharpness 越大越接近 min(x, hi)。
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    hi_val = float(hi) if not isinstance(hi, torch.Tensor) else hi
    return hi_val - torch.nn.functional.softplus(
        (hi_val - x) * sharpness, beta=1.0) / sharpness


def smooth_min(a: torch.Tensor, b, sharpness: float = 10.0) -> torch.Tensor:
    """平滑 min(a, b)。"""
    return smooth_upper_bound(a, b, sharpness)


def smooth_sign(x: torch.Tensor, temp: float = 0.01) -> torch.Tensor:
    """tanh 平滑 sign。"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    return torch.tanh(x / temp)


def smooth_step(x: torch.Tensor, threshold, temp: float = 1.0) -> torch.Tensor:
    """sigmoid 平滑阶跃函数。"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    threshold = float(threshold) if not isinstance(threshold, torch.Tensor) else threshold
    return torch.sigmoid((x - threshold) / temp)


# ── 基础运算（含 differentiable 开关） ─────────────────

def clamp(x, lo, hi, differentiable: bool = False, temp: float = 0.1):
    """限幅。differentiable=True 时使用 smooth_clamp。"""
    if differentiable:
        return smooth_clamp(x, lo, hi, temp)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    lo_val = float(lo) if not isinstance(lo, torch.Tensor) else lo
    hi_val = float(hi) if not isinstance(hi, torch.Tensor) else hi
    return torch.clamp(x, lo_val, hi_val)


def sign(x, differentiable: bool = False, temp: float = 0.01):
    """符号函数。differentiable=True 时使用 smooth_sign。"""
    if differentiable:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(float(x))
        return smooth_sign(x, temp)
    if isinstance(x, torch.Tensor):
        val = x.item()
    else:
        val = float(x)
    if abs(val) < 1e-3:
        return torch.tensor(0.0)
    return torch.tensor(1.0 if val > 0 else -1.0)


def rate_limit(prev, target, rate, dt, differentiable: bool = False,
               temp: float = 0.1):
    """速率限制器。rate 单位与 target/dt 相同。
    differentiable=True 时使用 straight-through estimator：
    前向走硬 clamp，反向梯度穿透（直接传到 target）。
    """
    if not isinstance(prev, torch.Tensor):
        prev = torch.tensor(float(prev))
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(float(target))
    max_delta = rate * dt
    delta = target - prev
    if differentiable:
        clamped_delta = _straight_through_clamp(delta, -max_delta, max_delta)
    else:
        clamped_delta = torch.clamp(delta, -max_delta, max_delta)
    return prev + clamped_delta


class _StraightThroughClamp(torch.autograd.Function):
    """Straight-through estimator for clamp.
    前向：硬 clamp。
    反向：梯度无条件传给 x；当 x<=lo 时也传给 lo，当 x>=hi 时也传给 hi。
    确保 lo/hi 来自 nn.Parameter 时梯度可流过。
    """
    @staticmethod
    def forward(ctx, x, lo, hi):
        lo_t = torch.as_tensor(lo) if not isinstance(lo, torch.Tensor) else lo
        hi_t = torch.as_tensor(hi) if not isinstance(hi, torch.Tensor) else hi
        ctx.save_for_backward(x, lo_t, hi_t)
        return torch.clamp(x, lo_t.item(), hi_t.item())

    @staticmethod
    def backward(ctx, grad_output):
        x, lo, hi = ctx.saved_tensors
        # x 总是收到梯度（straight-through）
        grad_x = grad_output
        # lo 在 x <= lo 时收到梯度（前向输出 = lo）
        grad_lo = grad_output if x.item() <= lo.item() else None
        # hi 在 x >= hi 时收到梯度（前向输出 = hi）
        grad_hi = grad_output if x.item() >= hi.item() else None
        return grad_x, grad_lo, grad_hi


def _straight_through_clamp(x, lo, hi):
    """Straight-through clamp: 前向硬限幅，反向梯度直通。
    lo/hi 可为 float 或 tensor（含 nn.Parameter）。
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x))
    if not isinstance(lo, torch.Tensor):
        lo = torch.tensor(float(lo))
    if not isinstance(hi, torch.Tensor):
        hi = torch.tensor(float(hi))
    return _StraightThroughClamp.apply(x, lo, hi)


def normalize_angle(angle) -> torch.Tensor:
    """归一化角度到 (-pi, pi]。使用 atan2 保证可微。"""
    if not isinstance(angle, torch.Tensor):
        angle = torch.tensor(float(angle))
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class PID:
    """PID 控制器，含抗积分饱和。按 spec §1.5。
    V2: 参数从外部传入，状态为 torch.Tensor。
    """
    def __init__(self):
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)

    def control(self, error, dt, kp, ki, kd,
                integrator_enable: bool, sat,
                differentiable: bool = False):
        """计算 PID 输出。kp/ki/kd/sat 从外部传入（可为 nn.Parameter）。"""
        if not isinstance(error, torch.Tensor):
            error = torch.tensor(float(error))
        if integrator_enable:
            self.integral = clamp(self.integral + error * dt, -sat, sat,
                                  differentiable)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error.detach().clone()
        return kp * error + ki * self.integral + kd * derivative

    def reset(self):
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)


class IIR:
    """一阶 IIR 低通滤波器（传递函数形式）。y = x - alpha * y_prev"""
    def __init__(self, alpha):
        if isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            self.alpha = torch.tensor(float(alpha))
        self.y_prev = torch.tensor(0.0)

    def update(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(float(x))
        y = x - self.alpha * self.y_prev
        self.y_prev = y.detach().clone()
        return y

    def reset(self):
        self.y_prev = torch.tensor(0.0)


@dataclass
class TrajectoryPoint:
    """轨迹点。保持 float 字段（不在梯度路径上）。"""
    x: float
    y: float
    theta: float
    kappa: float
    v: float
    a: float
    s: float
    t: float
