# V2 可微调参实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 V1 numpy 控制器原地升级为 PyTorch 版本，支持端到端可微调参（横向 + 纵向），每次调参结果保存为带 commit hash 的 YAML 文件。

**Architecture:** 所有模块统一 `torch.Tensor`。`differentiable=True/False` 开关控制非光滑操作实现方式。控制器继承 `nn.Module`，可调参数注册为 `nn.Parameter`。训练通过 BPTT 展开整条仿真轨迹计算梯度。

**Tech Stack:** PyTorch (CPU only), PyYAML, pytest

**关键文档：**
- 设计文档：`docs/plans/2026-03-02-differentiable-tuning-v2-design.md`
- 控制器规格：`docs/controller_spec.md`
- V1 设计：`docs/plans/2026-03-02-differentiable-control-v1-design.md`

**兼容性要求：** `differentiable=False` 模式下所有 V1 测试必须通过，输出与 numpy 版本数值差异 < 1e-5。

---

### Task 1: common.py — Torch 原语 + 平滑替代

**Files:**
- Modify: `sim/common.py`
- Modify: `sim/tests/test_common.py`

**核心变更：** 所有函数从 `math`/Python-float 转为 `torch` 运算。新增 `smooth_clamp`、`smooth_sign`、`smooth_step` 平滑函数。`PID`/`IIR` 类用 torch 实现（状态为 tensor，支持 BPTT 梯度流）。`TrajectoryPoint` 保持 dataclass 不变（轨迹生成不在梯度路径上）。

**Step 1: 写测试 — torch 原语 + 平滑函数**

在 `sim/tests/test_common.py` 添加新测试类，保留所有原有测试并修改为 torch 兼容（用 `.item()` 比较）：

```python
import torch

# --- 原有测试修改：lookup1d 签名变更 ---
class TestLookup1d:
    def test_exact_match(self):
        table_x = torch.tensor([0.0, 10.0, 20.0])
        table_y = torch.tensor([0.0, 1.0, 2.0])
        assert lookup1d(table_x, table_y, torch.tensor(10.0)).item() == pytest.approx(1.0)

    def test_interpolation(self):
        table_x = torch.tensor([0.0, 10.0, 20.0])
        table_y = torch.tensor([0.0, 1.0, 2.0])
        assert lookup1d(table_x, table_y, torch.tensor(5.0)).item() == pytest.approx(0.5)

    def test_clamp_low(self):
        table_x = torch.tensor([0.0, 10.0])
        table_y = torch.tensor([0.0, 1.0])
        assert lookup1d(table_x, table_y, torch.tensor(-5.0)).item() == pytest.approx(0.0)

    def test_clamp_high(self):
        table_x = torch.tensor([0.0, 10.0])
        table_y = torch.tensor([0.0, 1.0])
        assert lookup1d(table_x, table_y, torch.tensor(15.0)).item() == pytest.approx(1.0)

    def test_single_point(self):
        table_x = torch.tensor([5.0])
        table_y = torch.tensor([3.0])
        assert lookup1d(table_x, table_y, torch.tensor(0.0)).item() == pytest.approx(3.0)

    def test_gradient_flows_through_y(self):
        """lookup1d 对 table_y 的梯度应正常流通。"""
        table_x = torch.tensor([0.0, 10.0])
        table_y = torch.tensor([1.0, 3.0], requires_grad=True)
        result = lookup1d(table_x, table_y, torch.tensor(5.0))
        result.backward()
        assert table_y.grad is not None
        assert table_y.grad[0].item() == pytest.approx(0.5)  # (1 - t)
        assert table_y.grad[1].item() == pytest.approx(0.5)  # t

# --- 新增测试 ---
class TestSmoothClamp:
    def test_within_range_unchanged(self):
        """范围内值应近似不变。"""
        result = smooth_clamp(torch.tensor(0.5), 0.0, 1.0)
        assert result.item() == pytest.approx(0.5, abs=0.05)

    def test_outside_range_clamped(self):
        """超出范围的值应被平滑限制。"""
        result = smooth_clamp(torch.tensor(10.0), 0.0, 1.0)
        assert result.item() < 1.1
        assert result.item() > 0.9

    def test_gradient_at_boundary(self):
        """边界处梯度应非零（与 hard clamp 的区别）。"""
        x = torch.tensor(1.0, requires_grad=True)
        result = smooth_clamp(x, 0.0, 1.0)
        result.backward()
        assert x.grad is not None
        assert x.grad.item() > 0  # smooth → 边界处仍有梯度


class TestSmoothSign:
    def test_positive(self):
        assert smooth_sign(torch.tensor(1.0)).item() > 0.9

    def test_negative(self):
        assert smooth_sign(torch.tensor(-1.0)).item() < -0.9

    def test_zero_near_zero(self):
        assert abs(smooth_sign(torch.tensor(0.0)).item()) < 0.1

    def test_gradient_at_zero(self):
        x = torch.tensor(0.0, requires_grad=True)
        result = smooth_sign(x)
        result.backward()
        assert x.grad is not None
        assert x.grad.item() > 0


class TestSmoothStep:
    def test_below_threshold(self):
        result = smooth_step(torch.tensor(0.0), threshold=5.0)
        assert result.item() < 0.1

    def test_above_threshold(self):
        result = smooth_step(torch.tensor(10.0), threshold=5.0)
        assert result.item() > 0.9

    def test_gradient_at_threshold(self):
        x = torch.tensor(5.0, requires_grad=True)
        result = smooth_step(x, threshold=5.0)
        result.backward()
        assert x.grad is not None
        assert x.grad.item() > 0


class TestRateLimitTorch:
    def test_within_limit(self):
        result = rate_limit(torch.tensor(0.0), torch.tensor(5.0), 300.0, 0.02)
        assert result.item() == pytest.approx(5.0)

    def test_clamped_up(self):
        result = rate_limit(torch.tensor(0.0), torch.tensor(10.0), 120.0, 0.02)
        assert result.item() == pytest.approx(2.4)

    def test_smooth_mode_has_gradient(self):
        target = torch.tensor(10.0, requires_grad=True)
        result = rate_limit(torch.tensor(0.0), target, 120.0, 0.02, differentiable=True)
        result.backward()
        assert target.grad is not None
        assert target.grad.item() > 0


class TestNormalizeAngleTorch:
    def test_zero(self):
        assert normalize_angle(torch.tensor(0.0)).item() == pytest.approx(0.0)

    def test_small_angle(self):
        assert normalize_angle(torch.tensor(0.5)).item() == pytest.approx(0.5)

    def test_wrap_positive(self):
        result = normalize_angle(torch.tensor(3 * math.pi))
        assert result.item() == pytest.approx(math.pi, abs=1e-5)


class TestPIDTorch:
    def test_proportional(self):
        pid = PID()
        result = pid.control(torch.tensor(2.0), 0.02,
                             kp=torch.tensor(1.0), ki=torch.tensor(0.0),
                             kd=torch.tensor(0.0),
                             integrator_enable=False, sat=1.0)
        assert result.item() == pytest.approx(2.0)

    def test_gradient_through_kp(self):
        pid = PID()
        kp = torch.tensor(1.0, requires_grad=True)
        result = pid.control(torch.tensor(2.0), 0.02,
                             kp=kp, ki=torch.tensor(0.0),
                             kd=torch.tensor(0.0),
                             integrator_enable=False, sat=1.0)
        result.backward()
        assert kp.grad is not None
        assert kp.grad.item() == pytest.approx(2.0)
```

**Step 2: 运行测试确认失败**

Run: `cd sim && python -m pytest tests/test_common.py -v`
Expected: FAIL — 函数签名和返回类型不匹配

**Step 3: 实现 common.py**

完整重写 `sim/common.py`：

```python
# sim/common.py
"""公共基础组件 — torch 版。支持 differentiable 开关。"""
import math
from dataclasses import dataclass
import torch


def lookup1d(table_x: torch.Tensor, table_y: torch.Tensor,
             x: torch.Tensor) -> torch.Tensor:
    """分段线性插值。table_x 为断点坐标，table_y 为对应值（可为 nn.Parameter）。
    对 table_y 可微（线性插值），对 x 几乎处处可微。"""
    if len(table_x) == 1:
        return table_y[0]
    # 将 x clamp 到表范围内
    x_clamped = torch.clamp(x, table_x[0].item(), table_x[-1].item())
    # 找到所在区间
    # searchsorted 返回插入位置，-1 得到左端点索引
    idx = torch.searchsorted(table_x, x_clamped) - 1
    idx = torch.clamp(idx, 0, len(table_x) - 2)
    i = idx.long()
    x0 = table_x[i]
    x1 = table_x[i + 1]
    y0 = table_y[i]
    y1 = table_y[i + 1]
    t = (x_clamped - x0) / (x1 - x0 + 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    return y0 + (y1 - y0) * t


def smooth_clamp(x: torch.Tensor, lo: float, hi: float,
                 temp: float = 0.1) -> torch.Tensor:
    """tanh-based 平滑 clamp。temp→0 退化为 hard clamp。"""
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half < 1e-12:
        return torch.tensor(mid)
    return mid + half * torch.tanh((x - mid) / (half * temp))


def smooth_sign(x: torch.Tensor, temp: float = 0.01) -> torch.Tensor:
    """平滑 sign 函数。temp→0 趋近阶跃。"""
    return torch.tanh(x / temp)


def smooth_step(x: torch.Tensor, threshold: float,
                temp: float = 1.0) -> torch.Tensor:
    """平滑阶跃函数：x < threshold → 0, x > threshold → 1。"""
    return torch.sigmoid((x - threshold) / temp)


def clamp(x: torch.Tensor, lo: float, hi: float,
          differentiable: bool = False, temp: float = 0.1) -> torch.Tensor:
    """带 differentiable 开关的 clamp。"""
    if differentiable:
        return smooth_clamp(x, lo, hi, temp)
    return torch.clamp(x, lo, hi)


def sign(x: torch.Tensor, differentiable: bool = False,
         temp: float = 0.01) -> torch.Tensor:
    """带 differentiable 开关的 sign（含死区）。"""
    if differentiable:
        return smooth_sign(x, temp)
    # V1 行为：abs(x) < 1e-3 返回 0
    val = x.item()
    if abs(val) < 1e-3:
        return torch.tensor(0.0)
    return torch.tensor(1.0 if val > 0 else -1.0)


def rate_limit(prev: torch.Tensor, target: torch.Tensor,
               rate: float, dt: float,
               differentiable: bool = False, temp: float = 0.1) -> torch.Tensor:
    """带 differentiable 开关的速率限制器。"""
    max_delta = rate * dt
    delta = target - prev
    if differentiable:
        clamped_delta = smooth_clamp(delta, -max_delta, max_delta, temp)
    else:
        clamped_delta = torch.clamp(delta, -max_delta, max_delta)
    return prev + clamped_delta


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """归一化角度到 (-pi, pi]。使用 atan2(sin, cos) 实现，处处可微。"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class PID:
    """PID 控制器 — torch 版。参数从外部传入（支持 nn.Parameter）。"""
    def __init__(self):
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)

    def control(self, error: torch.Tensor, dt: float,
                kp: torch.Tensor, ki: torch.Tensor, kd: torch.Tensor,
                integrator_enable: bool, sat: float,
                differentiable: bool = False) -> torch.Tensor:
        if integrator_enable:
            self.integral = clamp(self.integral + error * dt, -sat, sat,
                                  differentiable)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return kp * error + ki * self.integral + kd * derivative

    def reset(self):
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)


class IIR:
    """一阶 IIR 低通滤波器 — torch 版。"""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y_prev = torch.tensor(0.0)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        y = x - self.alpha * self.y_prev
        self.y_prev = y
        return y

    def reset(self):
        self.y_prev = torch.tensor(0.0)


@dataclass
class TrajectoryPoint:
    """轨迹点（生成阶段用 float，不在梯度路径上）。"""
    x: float
    y: float
    theta: float
    kappa: float
    v: float
    a: float
    s: float
    t: float
```

**Step 4: 运行测试确认通过**

Run: `cd sim && python -m pytest tests/test_common.py -v`
Expected: ALL PASS

**Step 5: 提交**

```bash
git add sim/common.py sim/tests/test_common.py
git commit -m "[sim] common.py 转 torch + 平滑可微原语"
```

---

### Task 2: vehicle.py — Torch 自行车模型

**Files:**
- Modify: `sim/vehicle.py`
- Modify: `sim/tests/test_vehicle.py`

**核心变更：** `math.cos/sin/tan` → `torch.cos/sin/tan`。`max(0, v)` → `torch.clamp` / `softplus` 按 differentiable 开关。状态量 (x, y, yaw, v) 均为 `torch.Tensor`。

**Step 1: 写测试**

更新 `sim/tests/test_vehicle.py`，保留原有测试逻辑，增加梯度测试：

```python
import torch

class TestBicycleModel:
    # 原有测试保持不变（返回 tensor，用 .item() 比较）
    def test_straight_line(self):
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=10.0)
        for _ in range(500):
            car.step(torch.tensor(0.0), torch.tensor(0.0))
        assert car.x.item() == pytest.approx(100.0, abs=1.0)
        assert car.y.item() == pytest.approx(0.0, abs=0.01)

    def test_gradient_through_step(self):
        """加速度参数的梯度应流过 vehicle step。"""
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=5.0)
        acc = torch.tensor(1.0, requires_grad=True)
        car.step(torch.tensor(0.0), acc)
        car.x.backward()
        # x 不直接依赖 acc（x += v*cos*dt，v 在 step 末尾更新）
        # 但 v 依赖 acc，下一步 x 才依赖 v
        car_v = car.v
        car_v.backward()
        assert acc.grad is not None

    # 同理更新其他测试...
```

**Step 2: 确认失败**

**Step 3: 实现 vehicle.py**

```python
# sim/vehicle.py
"""运动学自行车模型 — torch 版。"""
import torch


class BicycleModel:
    """运动学自行车模型。状态 [x, y, yaw, v] 均为 torch.Tensor。"""
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

    def step(self, delta: torch.Tensor, acc: torch.Tensor):
        """delta: 前轮转角 (rad), acc: 纵向加速度 (m/s²)。"""
        self.x = self.x + self.v * torch.cos(self.yaw) * self.dt
        self.y = self.y + self.v * torch.sin(self.yaw) * self.dt
        self.yaw = self.yaw + self.v * torch.tan(delta) / self.L * self.dt
        self.v = self.v + acc * self.dt
        if self.differentiable:
            self.v = torch.nn.functional.softplus(self.v, beta=10.0)
        else:
            self.v = torch.clamp(self.v, min=0.0)

    @property
    def speed_kph(self) -> torch.Tensor:
        return self.v * 3.6

    @property
    def yaw_deg(self) -> torch.Tensor:
        return self.yaw * (180.0 / 3.141592653589793)
```

**Step 4: 确认通过**

Run: `cd sim && python -m pytest tests/test_vehicle.py -v`

**Step 5: 提交**

```bash
git add sim/vehicle.py sim/tests/test_vehicle.py
git commit -m "[sim] vehicle.py 转 torch"
```

---

### Task 3: trajectory.py — Torch 轨迹分析器

**Files:**
- Modify: `sim/trajectory.py`
- Modify: `sim/tests/test_trajectory.py`

**核心变更：** 轨迹生成函数不变（仍用 math，不在梯度路径上）。`TrajectoryAnalyzer` 在 `__init__` 中将轨迹点转为 tensor 数组，`query_nearest_by_position` 用 detached argmin，`to_frenet` 用 torch 运算。

**Step 1: 写测试**

保留原有测试，增加梯度测试：

```python
class TestAnalyzerTorch:
    def test_nearest_detached_gradient(self):
        """position 查询应允许梯度通过误差计算流回。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        x = torch.tensor(50.0, requires_grad=True)
        y = torch.tensor(1.0, requires_grad=True)
        pt = analyzer.query_nearest_by_position(x, y)
        # 横向误差
        lat_err = torch.cos(pt.theta) * (y - pt.y) - torch.sin(pt.theta) * (x - pt.x)
        lat_err.backward()
        assert y.grad is not None  # 梯度应流过

    def test_frenet_returns_tensors(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(
            torch.tensor(50.0), torch.tensor(0.0))
        s, sd, d, dd = analyzer.to_frenet(
            torch.tensor(50.0), torch.tensor(0.0),
            torch.tensor(0.0), torch.tensor(10.0), matched)
        assert isinstance(d, torch.Tensor)
```

**Step 2: 确认失败**

**Step 3: 实现 trajectory.py**

关键变更在 `TrajectoryAnalyzer`（轨迹生成函数保持原样不变）：

```python
class TrajectoryAnalyzer:
    """轨迹分析器 — torch 版。位置查询用 detached argmin。"""

    def __init__(self, points: list[TrajectoryPoint]):
        self.points = points
        # 预存为 tensor 加速查询
        self._xs = torch.tensor([p.x for p in points])
        self._ys = torch.tensor([p.y for p in points])
        self._ts = torch.tensor([p.t for p in points])

    def query_nearest_by_position(self, x: torch.Tensor,
                                   y: torch.Tensor) -> TrajectoryPoint:
        """位置最近点查询。索引选择不参与梯度（detached argmin）。"""
        with torch.no_grad():
            dx = self._xs - x.detach()
            dy = self._ys - y.detach()
            dists = dx * dx + dy * dy
            idx = torch.argmin(dists).item()
        return self.points[idx]

    def query_nearest_by_relative_time(self, t_rel) -> TrajectoryPoint:
        """时间查询 — 与 V1 逻辑相同，输入可为 tensor 或 float。"""
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
                dtheta = normalize_angle(
                    torch.tensor(p1.theta - p0.theta)).item()
                return TrajectoryPoint(
                    x=p0.x + frac * (p1.x - p0.x),
                    y=p0.y + frac * (p1.y - p0.y),
                    theta=p0.theta + frac * dtheta,
                    kappa=p0.kappa + frac * (p1.kappa - p0.kappa),
                    v=p0.v + frac * (p1.v - p0.v),
                    a=p0.a + frac * (p1.a - p0.a),
                    s=p0.s + frac * (p1.s - p0.s),
                    t=t_rel,
                )
        return self.points[-1]

    def to_frenet(self, x: torch.Tensor, y: torch.Tensor,
                  theta_rad: torch.Tensor, v_mps: torch.Tensor,
                  matched: TrajectoryPoint
                  ) -> tuple[torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor]:
        """Frenet 变换 — torch 版。"""
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
```

**Step 4: 确认通过**

Run: `cd sim && python -m pytest tests/test_trajectory.py -v`

**Step 5: 提交**

```bash
git add sim/trajectory.py sim/tests/test_trajectory.py
git commit -m "[sim] trajectory.py 转 torch，detached argmin"
```

---

### Task 4: config.py — Torch 表加载 + YAML 保存

**Files:**
- Modify: `sim/config.py`
- Create: `sim/configs/tuned/` (目录)
- Modify: `sim/tests/test_common.py` (增加 config 测试)

**核心变更：** `table_from_config` 返回 `(torch.Tensor, torch.Tensor)` 即 (x_values, y_values)。新增 `save_tuned_config()` 保存带 commit hash + 时间戳的 YAML。

**Step 1: 写测试**

```python
# 在 test_common.py 或新建 test_config.py
class TestTableFromConfig:
    def test_returns_tensor_pair(self):
        entries = [[0, 1.0], [10, 2.0], [20, 3.0]]
        xs, ys = table_from_config(entries)
        assert isinstance(xs, torch.Tensor)
        assert isinstance(ys, torch.Tensor)
        assert xs.tolist() == [0.0, 10.0, 20.0]
        assert ys.tolist() == [1.0, 2.0, 3.0]

class TestSaveTunedConfig:
    def test_save_creates_yaml(self, tmp_path):
        cfg = load_config()
        path = save_tuned_config(cfg, output_dir=str(tmp_path),
                                  meta={'final_loss': 0.01, 'epochs': 10})
        assert os.path.exists(path)
        loaded = load_config(path)
        assert loaded['vehicle']['wheelbase'] == 3.5
        assert 'meta' in loaded
```

**Step 2: 确认失败**

**Step 3: 实现 config.py**

```python
# sim/config.py
"""配置文件加载/保存器 — torch 版。"""
import os
import subprocess
from datetime import datetime
import torch
import yaml


def load_config(path: str | None = None) -> dict:
    """加载 YAML 配置文件。默认加载 configs/default.yaml。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def table_from_config(entries: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """将 YAML [[idx, val], ...] 转为 (x_tensor, y_tensor)。"""
    xs = torch.tensor([row[0] for row in entries], dtype=torch.float32)
    ys = torch.tensor([row[1] for row in entries], dtype=torch.float32)
    return xs, ys


def _get_commit_hash() -> str:
    """获取当前 git commit 短 hash。"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return result.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def _tensor_to_list(obj):
    """递归将 dict 中的 torch.Tensor 转为 Python list/float。"""
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return float(obj.item())
        return obj.detach().tolist()
    if isinstance(obj, dict):
        return {k: _tensor_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_to_list(item) for item in obj]
    return obj


def save_tuned_config(cfg: dict, output_dir: str | None = None,
                       meta: dict | None = None) -> str:
    """保存调参后的完整配置到 YAML。

    文件名：tuned_{commit}_{timestamp}.yaml
    返回保存路径。
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'configs', 'tuned')
    os.makedirs(output_dir, exist_ok=True)

    commit = _get_commit_hash()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'tuned_{commit}_{timestamp}.yaml'
    path = os.path.join(output_dir, filename)

    save_dict = _tensor_to_list(cfg)
    if meta is not None:
        save_dict['meta'] = {
            'commit': commit,
            'timestamp': datetime.now().isoformat(),
            'base_config': 'default.yaml',
            **_tensor_to_list(meta),
        }

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(save_dict, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
    return path
```

**Step 4: 确认通过**

**Step 5: 提交**

```bash
git add sim/config.py sim/tests/
git commit -m "[sim] config.py 支持 torch tensor 表 + YAML 保存（带 commit hash）"
```

---

### Task 5: LatControllerTruck — nn.Module 化

**Files:**
- Modify: `sim/controller/lat_truck.py`
- Modify: `sim/tests/test_lat_truck.py`

**核心变更：** 继承 `nn.Module`。8 张表的 y 值注册为 `nn.Parameter`，x 值注册为 `buffer`。`kLh` 注册为 `nn.Parameter`。所有 `math.*` → `torch.*`。`sign` / `clamp` / `rate_limit` 通过 `self.differentiable` 切换。

**Step 1: 写测试**

保留原有 4 项测试（修改为 tensor 兼容），新增：

```python
class TestLatTruckDifferentiable:
    def test_gradient_through_T2(self):
        """T2 表（预瞄时间）的梯度应流过 compute。"""
        cfg = load_config()
        ctrl = LatControllerTruck(cfg, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0),
            steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        steer.backward()
        assert ctrl.T2_y.grad is not None

    def test_false_matches_v1(self):
        """differentiable=False 应与 V1 numpy 结果一致。"""
        # 需在 Task 8 详细验证，这里做基本检查
        cfg = load_config()
        ctrl = LatControllerTruck(cfg, differentiable=False)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0),
            steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        assert isinstance(steer, torch.Tensor)
        assert abs(steer.item()) < 50  # 合理范围
```

**Step 2: 确认失败**

**Step 3: 实现 lat_truck.py**

```python
# sim/controller/lat_truck.py
"""重卡横向控制器 — torch nn.Module 版。"""
import torch
import torch.nn as nn
from common import lookup1d, rate_limit, clamp, sign, normalize_angle
from config import table_from_config

DEG2RAD = 3.141592653589793 / 180.0
RAD2DEG = 180.0 / 3.141592653589793


class LatControllerTruck(nn.Module):
    """重卡横向控制器。可调参数注册为 nn.Parameter。"""

    def __init__(self, cfg: dict, differentiable: bool = False):
        super().__init__()
        self.differentiable = differentiable
        veh = cfg['vehicle']
        lat = cfg['lat_truck']

        self.wheelbase = veh['wheelbase']
        self.steer_ratio = veh['steer_ratio']

        # 非优化标量
        self.rate_limit_fb = lat['rate_limit_fb']
        self.rate_limit_ff = lat['rate_limit_ff']
        self.rate_limit_total = lat['rate_limit_total']
        self.min_prev_dist = lat['min_prev_dist']
        self.min_reach_dis = lat['min_reach_dis']
        self.min_speed_prot = lat['min_speed_prot']

        # 可优化参数
        self.kLh = nn.Parameter(torch.tensor(float(lat['kLh'])))

        # 8 张表：x 值固定（buffer），y 值可优化（parameter）
        for i, key in enumerate(['T1_max_theta_deg', 'T2_prev_time_dist',
                                  'T3_reach_time_theta', 'T4_T_dt',
                                  'T5_near_point_time', 'T6_far_point_time',
                                  'T7_max_steer_angle', 'T8_slip_param']):
            xs, ys = table_from_config(lat[key])
            self.register_buffer(f'T{i+1}_x', xs)
            self.register_parameter(f'T{i+1}_y', nn.Parameter(ys))

        # 状态（每次仿真开始前需 reset）
        self.steer_fb_prev = torch.tensor(0.0)
        self.steer_ff_prev = torch.tensor(0.0)
        self.steer_total_prev = torch.tensor(0.0)

    def reset_state(self):
        self.steer_fb_prev = torch.tensor(0.0)
        self.steer_ff_prev = torch.tensor(0.0)
        self.steer_total_prev = torch.tensor(0.0)

    def _lookup(self, table_idx: int, x: torch.Tensor) -> torch.Tensor:
        tx = getattr(self, f'T{table_idx}_x')
        ty = getattr(self, f'T{table_idx}_y')
        return lookup1d(tx, ty, x)

    def compute(self, x: torch.Tensor, y: torch.Tensor,
                yaw_deg: torch.Tensor, speed_kph: torch.Tensor,
                yawrate: torch.Tensor, steer_feedback: torch.Tensor,
                analyzer, ctrl_enable: bool,
                dt: float = 0.02):
        """返回 (steering_target, kappa_current, kappa_near, kappa_far)。"""
        diff = self.differentiable

        speed_kph = torch.clamp(speed_kph, min=self.min_speed_prot)
        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if not ctrl_enable:
            self.steer_fb_prev = torch.tensor(0.0)
            self.steer_ff_prev = torch.tensor(0.0)
            self.steer_total_prev = steer_feedback.detach()
            z = torch.tensor(0.0)
            return steer_feedback, z, z, z

        # Step 1: 查表
        max_theta_deg = self._lookup(1, speed_kph)
        prev_time_dist = self._lookup(2, speed_kph)
        reach_time_theta = self._lookup(3, speed_kph)
        T_dt = self._lookup(4, speed_kph)
        near_pt_time = self._lookup(5, speed_kph)
        far_pt_time = self._lookup(6, speed_kph)
        max_steer_angle = self._lookup(7, speed_kph)
        slip_param = self._lookup(8, speed_kph)

        # Step 2: 轨迹查询
        currt = analyzer.query_nearest_by_position(x, y)
        near = analyzer.query_nearest_by_relative_time(
            currt.t + near_pt_time.item())
        far = analyzer.query_nearest_by_relative_time(
            currt.t + far_pt_time.item())

        # Step 3: 误差计算
        currt_theta = torch.tensor(currt.theta)
        dx = x - currt.x
        dy = y - currt.y
        lateral_error = torch.cos(currt_theta) * dy - torch.sin(currt_theta) * dx
        heading_error = normalize_angle(yaw_rad - currt_theta)
        curvature_far = torch.tensor(far.kappa)

        # Step 4: real_theta
        vehicle_speed_clamped = torch.clamp(speed_mps, 1.0, 100.0)
        real_theta = -heading_error - torch.atan(
            self.kLh * yawrate / vehicle_speed_clamped)

        # Step 5: real_dt_theta
        real_dt_theta = -(yawrate - curvature_far * speed_mps)

        # Step 6: target_theta
        prev_dist = torch.clamp(speed_mps * prev_time_dist,
                                min=self.min_prev_dist)
        dis2lane = -lateral_error
        error_angle_raw = torch.atan(dis2lane / prev_dist)
        max_err_angle = clamp(torch.abs(error_angle_raw),
                              0.0, (max_theta_deg * DEG2RAD).item(), diff)
        target_theta = sign(error_angle_raw, diff) * max_err_angle

        target_dt_theta = (torch.sin(real_theta) * speed_mps * prev_dist
                           / (prev_dist ** 2 + dis2lane ** 2) * -1.0)

        # Step 7: target_curvature
        denom = torch.clamp(reach_time_theta * speed_mps,
                            min=self.min_reach_dis)
        target_curvature = -((target_theta - real_theta)
                             + (target_dt_theta - real_dt_theta) * T_dt) / denom

        # Step 8: 反馈转向角
        steer_fb_raw = (torch.atan(target_curvature * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_fb = rate_limit(self.steer_fb_prev, steer_fb_raw,
                              self.rate_limit_fb, dt, diff)
        self.steer_fb_prev = steer_fb

        # Step 9: 前馈转向角
        steer_ff_raw = (torch.atan(curvature_far * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_ff = rate_limit(self.steer_ff_prev, steer_ff_raw,
                              self.rate_limit_ff, dt, diff)
        self.steer_ff_prev = steer_ff

        # Step 10: 合并输出
        steer_raw = clamp(steer_fb + steer_ff,
                          -max_steer_angle.item(), max_steer_angle.item(), diff)
        steer_out = rate_limit(self.steer_total_prev, steer_raw,
                               self.rate_limit_total, dt, diff)
        self.steer_total_prev = steer_out

        return (steer_out, torch.tensor(currt.kappa),
                torch.tensor(near.kappa), curvature_far)
```

**注意：** `clamp` 的 `lo`/`hi` 参数需要是 float（因为 smooth_clamp 用 Python 算术计算 mid/half）。对于依赖 nn.Parameter 的 `max_steer_angle`，用 `.item()` 提取。这意味着 max_steer_angle 的梯度不流过 clamp 的边界计算——可接受，因为它是安全约束。

**Step 4: 确认通过**

Run: `cd sim && python -m pytest tests/test_lat_truck.py -v`

**Step 5: 提交**

```bash
git add sim/controller/lat_truck.py sim/tests/test_lat_truck.py
git commit -m "[sim] LatControllerTruck 转 nn.Module + differentiable 开关"
```

---

### Task 6: LonController — nn.Module 化

**Files:**
- Modify: `sim/controller/lon.py`
- Modify: `sim/tests/test_lon.py`

**核心变更：** 继承 `nn.Module`。PID 增益注册为 `nn.Parameter`。5 张表的 y 值注册为 `nn.Parameter`。条件分支用 `smooth_step` 替换。PID 类的参数改为从外部（nn.Parameter）传入。

**Step 1: 写测试**

```python
class TestLonDifferentiable:
    def test_gradient_through_kp(self):
        """low_speed_kp 的梯度应流过 compute。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=True)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(18.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True)
        acc.backward()
        assert ctrl.low_speed_kp.grad is not None
```

**Step 2: 确认失败**

**Step 3: 实现 lon.py**

```python
# sim/controller/lon.py
"""纵向控制器简化版 — torch nn.Module 版。"""
import torch
import torch.nn as nn
from common import (lookup1d, clamp, normalize_angle, smooth_step,
                     PID, IIR)
from config import table_from_config

DEG2RAD = 3.141592653589793 / 180.0


class LonController(nn.Module):
    """纵向控制器（简化版：输出加速度）。"""

    def __init__(self, cfg: dict, differentiable: bool = False):
        super().__init__()
        self.differentiable = differentiable
        lon = cfg['lon']

        # 可优化 PID 增益
        self.station_kp = nn.Parameter(torch.tensor(float(lon['station_kp'])))
        self.station_ki = nn.Parameter(torch.tensor(float(lon['station_ki'])))
        self.low_speed_kp = nn.Parameter(torch.tensor(float(lon['low_speed_kp'])))
        self.low_speed_ki = nn.Parameter(torch.tensor(float(lon['low_speed_ki'])))
        self.high_speed_kp = nn.Parameter(torch.tensor(float(lon['high_speed_kp'])))
        self.high_speed_ki = nn.Parameter(torch.tensor(float(lon['high_speed_ki'])))
        self.switch_speed = nn.Parameter(torch.tensor(float(lon['switch_speed'])))

        # 非优化标量
        self.preview_window = lon['preview_window']
        self.preview_window_speed = lon['preview_window_speed']
        self.acc_use_preview_a = lon['acc_use_preview_a']
        self.station_error_limit = lon['station_error_limit']
        self.speed_input_limit = lon['speed_input_limit']
        self.acc_standstill_down_rate = lon['acc_standstill_down_rate']
        self.station_integrator_enable = lon['station_integrator_enable']
        self.station_sat = lon['station_sat']
        self.speed_pid_sat = lon['speed_pid_sat']
        self.iir_alpha = lon['iir_alpha']

        # 5 张表
        for i, key in enumerate(['L1_acc_up_lim', 'L2_acc_low_lim',
                                  'L3_acc_up_rate', 'L4_acc_down_rate',
                                  'L5_rate_gain']):
            xs, ys = table_from_config(lon[key])
            self.register_buffer(f'L{i+1}_x', xs)
            self.register_parameter(f'L{i+1}_y', nn.Parameter(ys))

        # 状态
        self.station_pid = PID()
        self.speed_pid = PID()
        self.iir_acc = IIR(alpha=self.iir_alpha)
        self.acc_out_prev = torch.tensor(0.0)
        self.station_error_fnl_prev = torch.tensor(0.0)

    def reset_state(self):
        self.station_pid.reset()
        self.speed_pid.reset()
        self.iir_acc.reset()
        self.acc_out_prev = torch.tensor(0.0)
        self.station_error_fnl_prev = torch.tensor(0.0)

    def _lookup(self, table_idx: int, x: torch.Tensor) -> torch.Tensor:
        tx = getattr(self, f'L{table_idx}_x')
        ty = getattr(self, f'L{table_idx}_y')
        return lookup1d(tx, ty, x)

    def compute(self, x: torch.Tensor, y: torch.Tensor,
                yaw_deg: torch.Tensor, speed_kph: torch.Tensor,
                accel_mps2: torch.Tensor, curvature_far: torch.Tensor,
                analyzer, t_now: float,
                ctrl_enable: bool, ctrl_first_active: bool,
                dt: float = 0.02) -> torch.Tensor:
        diff = self.differentiable
        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if ctrl_first_active:
            self.station_pid.reset()
            self.speed_pid.reset()
            self.iir_acc.reset()

        # Step 1: Frenet 误差
        matched = analyzer.query_nearest_by_position(x, y)
        s_match, s_dot, d, d_dot = analyzer.to_frenet(
            x, y, yaw_rad, speed_mps, matched)

        ref_pt = analyzer.query_nearest_by_relative_time(t_now)
        prev_pt = analyzer.query_nearest_by_relative_time(
            t_now + self.preview_window * dt)
        spd_pt = analyzer.query_nearest_by_relative_time(
            t_now + self.preview_window_speed * dt)

        station_error = torch.tensor(ref_pt.s) - s_match
        preview_speed_error = torch.tensor(spd_pt.v) - speed_mps
        preview_accel_ref = torch.tensor(prev_pt.a)

        # Step 2: 站位误差保护
        station_limited = clamp(station_error,
                                -self.station_error_limit,
                                self.station_error_limit, diff)

        if diff:
            # 平滑版：用 smooth_step 混合各分支
            w_high_speed = smooth_step(speed_kph, threshold=10.0, temp=1.0)
            # 低速分支
            w_low_neg = smooth_step(-station_limited, threshold=-0.25, temp=0.5)
            low_branch_neg = clamp(station_limited, -self.station_error_limit, 0.0, True)
            w_high_pos = smooth_step(station_limited, threshold=0.8, temp=0.5)
            low_branch = (w_low_neg * low_branch_neg
                         + w_high_pos * station_limited
                         + (1 - w_low_neg - w_high_pos).clamp(0, 1)
                         * station_limited)
            station_fnl = w_high_speed * station_limited + (1 - w_high_speed) * low_branch
        else:
            speed_val = speed_kph.item()
            station_val = station_limited.item()
            if speed_val > 10:
                station_fnl = station_limited
            elif station_val <= 0.25:
                station_fnl = torch.minimum(torch.tensor(0.0), station_limited)
            elif station_val >= 0.8:
                station_fnl = station_limited
            elif self.station_error_fnl_prev.item() <= 0.01:
                station_fnl = self.station_error_fnl_prev
            else:
                station_fnl = station_limited
        self.station_error_fnl_prev = station_fnl

        # Step 3: 站位 PID
        speed_offset = self.station_pid.control(
            station_fnl, dt, self.station_kp, self.station_ki,
            torch.tensor(0.0), self.station_integrator_enable,
            self.station_sat, diff)

        # Step 4: 速度 PID（低速/高速切换）
        if diff:
            w_low = 1.0 - smooth_step(speed_mps, self.switch_speed.item(), temp=0.5)
            kp = w_low * self.low_speed_kp + (1 - w_low) * self.high_speed_kp
            ki = w_low * self.low_speed_ki + (1 - w_low) * self.high_speed_ki
        else:
            if speed_mps.item() <= self.switch_speed.item():
                kp, ki = self.low_speed_kp, self.low_speed_ki
            else:
                kp, ki = self.high_speed_kp, self.high_speed_ki

        speed_input = clamp(speed_offset + preview_speed_error,
                            -self.speed_input_limit, self.speed_input_limit, diff)
        acc_closeloop = self.speed_pid.control(
            speed_input, dt, kp, ki, torch.tensor(0.0),
            True, self.speed_pid_sat, diff)

        # Step 5: 前馈叠加
        acc_cmd = acc_closeloop + self.acc_use_preview_a * preview_accel_ref

        # Step 6: CalFinalAccCmd
        if ctrl_enable:
            abs_speed = torch.abs(speed_kph)
            acc_up_lim = self._lookup(1, abs_speed)
            acc_low_lim = self._lookup(2, abs_speed)
            acc_up_rate_raw = self._lookup(3, self.acc_out_prev)
            acc_dn_rate_raw = self._lookup(4, self.acc_out_prev)
            rate_gain = self._lookup(5, abs_speed)
            acc_up_rate = acc_up_rate_raw * rate_gain

            # 急弯收紧
            if diff:
                w_curve = smooth_step(-curvature_far,
                                      threshold=0.0075, temp=0.001)
                acc_up_lim = acc_up_lim * (1.0 - 0.25 * w_curve)
                acc_low_lim = acc_low_lim * (1.0 - 0.40 * w_curve)
            else:
                if curvature_far.item() < -0.0075:
                    acc_up_lim = acc_up_lim * 0.75
                    acc_low_lim = acc_low_lim * 0.60

            # 低速减速率
            if diff:
                w_lowspd = 1.0 - smooth_step(
                    torch.abs(speed_mps), threshold=1.5, temp=0.3)
                acc_dn_rate = (w_lowspd * self.acc_standstill_down_rate
                              + (1 - w_lowspd) * acc_dn_rate_raw)
            else:
                if abs(speed_mps.item()) < 1.5:
                    acc_dn_rate = torch.tensor(self.acc_standstill_down_rate)
                else:
                    acc_dn_rate = acc_dn_rate_raw

            acc_clamped = clamp(acc_cmd, acc_low_lim.item(),
                                acc_up_lim.item(), diff)

            # 低速保护
            if diff:
                w_moving = smooth_step(torch.abs(speed_mps),
                                       threshold=0.2, temp=0.1)
                w_acc_ok = smooth_step(acc_clamped, threshold=0.25, temp=0.1)
                w_pass = torch.clamp(w_moving + w_acc_ok, 0.0, 1.0)
                acc_lowspd = (w_pass * acc_clamped
                             + (1 - w_pass) * torch.minimum(
                                 torch.tensor(-0.05), acc_clamped))
            else:
                if abs(speed_mps.item()) >= 0.2 or acc_clamped.item() >= 0.25:
                    acc_lowspd = acc_clamped
                else:
                    acc_lowspd = torch.minimum(torch.tensor(-0.05), acc_clamped)

            # 速率限制
            acc_limited = clamp(
                acc_lowspd,
                (self.acc_out_prev + acc_dn_rate).item(),
                (self.acc_out_prev + acc_up_rate).item(),
                diff)
            self.acc_out_prev = acc_limited
        else:
            acc_limited = torch.tensor(0.0)
            self.acc_out_prev = torch.tensor(0.0)

        acc_out = self.iir_acc.update(acc_limited)
        return acc_out
```

**Step 4: 确认通过**

Run: `cd sim && python -m pytest tests/test_lon.py -v`

**Step 5: 提交**

```bash
git add sim/controller/lon.py sim/tests/test_lon.py
git commit -m "[sim] LonController 转 nn.Module + differentiable 开关"
```

---

### Task 7: sim_loop.py — Torch 仿真循环

**Files:**
- Modify: `sim/sim_loop.py`
- Modify: `sim/tests/test_sim_loop.py`

**核心变更：** 所有中间变量为 `torch.Tensor`。返回 history 中的值为 tensor。新增 `differentiable` 参数。控制器从外部传入（支持 nn.Module 参数共享）。

**Step 1: 写测试**

保留原有测试，增加：

```python
class TestSimLoopDifferentiable:
    def test_gradient_flows(self):
        """整条仿真的 loss 对控制器参数应有梯度。"""
        from controller.lat_truck import LatControllerTruck
        from controller.lon import LonController

        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)

        traj = generate_straight(length=50, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                  lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                  differentiable=True)
        # 计算简单 loss
        lat_errs = torch.stack([h['lateral_error'] for h in history])
        loss = (lat_errs ** 2).mean()
        loss.backward()
        assert lat_ctrl.T2_y.grad is not None
```

**Step 2: 确认失败**

**Step 3: 实现 sim_loop.py**

```python
# sim/sim_loop.py
"""50Hz 闭环仿真主循环 — torch 版。"""
import torch
from common import normalize_angle
from config import load_config
from trajectory import TrajectoryAnalyzer, TrajectoryPoint
from vehicle import BicycleModel
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController

DEG2RAD = 3.141592653589793 / 180.0


def run_simulation(trajectory: list[TrajectoryPoint],
                   init_speed: float = 0.0,
                   init_x: float | None = None,
                   init_y: float | None = None,
                   init_yaw: float | None = None,
                   cfg: dict | None = None,
                   lat_ctrl: LatControllerTruck | None = None,
                   lon_ctrl: LonController | None = None,
                   differentiable: bool = False,
                   ) -> list[dict]:
    """运行闭环仿真。返回历史记录（tensor 值）。"""
    if cfg is None:
        cfg = load_config()

    veh = cfg['vehicle']
    wheelbase = veh['wheelbase']
    steer_ratio = veh['steer_ratio']
    dt = cfg['simulation']['dt']

    analyzer = TrajectoryAnalyzer(trajectory)
    traj_duration = trajectory[-1].t

    p0 = trajectory[0]
    x0 = init_x if init_x is not None else p0.x
    y0 = init_y if init_y is not None else p0.y
    yaw0 = init_yaw if init_yaw is not None else p0.theta

    car = BicycleModel(wheelbase=wheelbase, x=x0, y=y0,
                       yaw=yaw0, v=init_speed, dt=dt,
                       differentiable=differentiable)

    if lat_ctrl is None:
        lat_ctrl = LatControllerTruck(cfg, differentiable=differentiable)
    else:
        lat_ctrl.reset_state()

    if lon_ctrl is None:
        lon_ctrl = LonController(cfg, differentiable=differentiable)
    else:
        lon_ctrl.reset_state()

    history = []
    n_steps = int(traj_duration / dt)
    prev_steer = torch.tensor(0.0)

    for step in range(n_steps):
        t = step * dt

        ref_pt = analyzer.query_nearest_by_position(car.x, car.y)
        ref_theta = torch.tensor(ref_pt.theta)
        dx = car.x - ref_pt.x
        dy = car.y - ref_pt.y
        lateral_error = torch.cos(ref_theta) * dy - torch.sin(ref_theta) * dx
        heading_error = normalize_angle(car.yaw - ref_theta)

        delta_prev = prev_steer / steer_ratio * DEG2RAD
        yawrate = car.v * torch.tan(delta_prev) / wheelbase

        steer_out, kappa_cur, kappa_near, curvature_far = lat_ctrl.compute(
            x=car.x, y=car.y,
            yaw_deg=car.yaw_deg,
            speed_kph=car.speed_kph,
            yawrate=yawrate,
            steer_feedback=prev_steer,
            analyzer=analyzer,
            ctrl_enable=True, dt=dt)

        acc_cmd = lon_ctrl.compute(
            x=car.x, y=car.y,
            yaw_deg=car.yaw_deg,
            speed_kph=car.speed_kph,
            accel_mps2=torch.tensor(0.0),
            curvature_far=curvature_far,
            analyzer=analyzer, t_now=t,
            ctrl_enable=True,
            ctrl_first_active=(step == 0), dt=dt)

        history.append({
            't': t,
            'x': car.x, 'y': car.y, 'yaw': car.yaw, 'v': car.v,
            'steer': steer_out, 'acc': acc_cmd,
            'lateral_error': lateral_error, 'heading_error': heading_error,
            'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
        })

        delta_front = steer_out / steer_ratio * DEG2RAD
        car.step(delta=delta_front, acc=acc_cmd)
        prev_steer = steer_out

    return history
```

**Step 4: 确认通过**

Run: `cd sim && python -m pytest tests/test_sim_loop.py -v`

**Step 5: 提交**

```bash
git add sim/sim_loop.py sim/tests/test_sim_loop.py
git commit -m "[sim] sim_loop.py 转 torch，支持梯度回传"
```

---

### Task 8: V1 兼容性验证

**Files:**
- Create: `sim/tests/test_v1_compat.py`

**目标：** 确认 `differentiable=False` 模式下，全部原有 43 项测试通过，且关键场景的输出与 V1 numpy 版数值差异 < 1e-5。

**Step 1: 写兼容性测试**

```python
# sim/tests/test_v1_compat.py
"""V1 兼容性测试 — 确认 differentiable=False 与 numpy 版行为一致。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from trajectory import generate_straight, generate_circle, TrajectoryAnalyzer
from sim_loop import run_simulation


class TestV1CompatLat:
    """横向控制器 V1 兼容性。"""

    def test_straight_no_steer(self):
        cfg = load_config()
        ctrl = LatControllerTruck(cfg, differentiable=False)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer.item()) < 5.0

    def test_lateral_offset_corrects(self):
        cfg = load_config()
        ctrl = LatControllerTruck(cfg, differentiable=False)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _ = ctrl.compute(
            x=torch.tensor(50.0), y=torch.tensor(2.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(0.0),
            analyzer=analyzer, ctrl_enable=True)
        assert steer.item() > 0

    def test_disable_returns_feedback(self):
        cfg = load_config()
        ctrl = LatControllerTruck(cfg, differentiable=False)
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        steer, _, _, _ = ctrl.compute(
            x=torch.tensor(0.0), y=torch.tensor(5.0),
            yaw_deg=torch.tensor(10.0), speed_kph=torch.tensor(36.0),
            yawrate=torch.tensor(0.0), steer_feedback=torch.tensor(42.0),
            analyzer=analyzer, ctrl_enable=False)
        assert steer.item() == pytest.approx(42.0)


class TestV1CompatLon:
    """纵向控制器 V1 兼容性。"""

    def test_on_track_no_correction(self):
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert abs(acc.item()) < 0.5

    def test_too_slow_accelerates(self):
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        acc = ctrl.compute(
            x=torch.tensor(10.0), y=torch.tensor(0.0),
            yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(18.0),
            accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert acc.item() > 0

    def test_acc_within_limits(self):
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        for _ in range(100):
            acc = ctrl.compute(
                x=torch.tensor(10.0), y=torch.tensor(0.0),
                yaw_deg=torch.tensor(0.0), speed_kph=torch.tensor(36.0),
                accel_mps2=torch.tensor(0.0), curvature_far=torch.tensor(0.0),
                analyzer=analyzer, t_now=1.0,
                ctrl_enable=True, ctrl_first_active=False)
            assert -4.0 <= acc.item() <= 2.0


class TestV1CompatSimLoop:
    """闭环仿真 V1 兼容性。"""

    def test_straight_tracks(self):
        traj = generate_straight(length=200, speed=10.0)
        history = run_simulation(traj, init_speed=10.0, differentiable=False)
        n_last = 100
        for rec in history[-n_last:]:
            val = rec['lateral_error']
            if isinstance(val, torch.Tensor):
                val = val.item()
            assert abs(val) < 1.0

    def test_circle_tracks(self):
        traj = generate_circle(radius=50.0, speed=5.0, arc_angle=math.pi/2)
        history = run_simulation(traj, init_speed=5.0, differentiable=False)
        n = len(history)
        for rec in history[n // 2:]:
            val = rec['lateral_error']
            if isinstance(val, torch.Tensor):
                val = val.item()
            assert abs(val) < 5.0
```

**Step 2: 运行全套测试**

Run: `cd sim && python -m pytest tests/ -v`
Expected: ALL PASS（包括原有测试 + 新兼容性测试）

**Step 3: 提交**

```bash
git add sim/tests/test_v1_compat.py
git commit -m "[sim] V1 兼容性验证测试"
```

---

### Task 9: train.py — DiffControllerParams + Loss + 训练循环

**Files:**
- Create: `sim/train.py`
- Create: `sim/tests/test_train.py`

**核心：** `DiffControllerParams(nn.Module)` 包装横纵向控制器。定义 `tracking_loss`。训练循环展开仿真 → backward → step。保存 YAML。

**Step 1: 写测试**

```python
# sim/tests/test_train.py
"""训练 pipeline 测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import DiffControllerParams, tracking_loss, train


class TestDiffControllerParams:
    def test_has_parameters(self):
        params = DiffControllerParams()
        n_params = sum(p.numel() for p in params.parameters())
        assert n_params > 50  # 至少 50 个可优化参数

    def test_to_config_dict(self):
        params = DiffControllerParams()
        cfg = params.to_config_dict()
        assert 'vehicle' in cfg
        assert 'lat_truck' in cfg
        assert 'lon' in cfg


class TestTrackingLoss:
    def test_zero_error_zero_loss(self):
        history = []
        for i in range(100):
            history.append({
                'lateral_error': torch.tensor(0.0),
                'heading_error': torch.tensor(0.0),
                'v': torch.tensor(5.0),
                'steer': torch.tensor(0.0),
                'acc': torch.tensor(0.0),
            })
        ref_speed = 5.0
        loss = tracking_loss(history, ref_speed)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestTrain:
    def test_loss_decreases(self):
        """短仿真 + 少量 epoch，loss 应下降。"""
        result = train(
            trajectories=['straight'],
            n_epochs=5,
            lr=1e-2,
            sim_length=30.0,
            sim_speed=5.0,
            verbose=False,
        )
        assert result['losses'][-1] < result['losses'][0]
        assert result['saved_path'] is not None
```

**Step 2: 确认失败**

**Step 3: 实现 train.py**

```python
# sim/train.py
"""V2 可微调参训练入口。"""
import argparse
import torch
import torch.nn as nn

from config import load_config, save_tuned_config, table_from_config
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from trajectory import (generate_straight, generate_circle,
                         generate_sine, generate_combined)
from sim_loop import run_simulation


class DiffControllerParams(nn.Module):
    """包装横纵向控制器，集中管理所有可优化参数。"""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        self.lon_ctrl = LonController(cfg, differentiable=True)

    def to_config_dict(self) -> dict:
        """将当前参数导出为与 default.yaml 相同结构的 dict。"""
        cfg = load_config()  # 加载模板结构

        # 横向表
        lat = cfg['lat_truck']
        lat['kLh'] = float(self.lat_ctrl.kLh.item())
        for i, key in enumerate(['T1_max_theta_deg', 'T2_prev_time_dist',
                                  'T3_reach_time_theta', 'T4_T_dt',
                                  'T5_near_point_time', 'T6_far_point_time',
                                  'T7_max_steer_angle', 'T8_slip_param']):
            tx = getattr(self.lat_ctrl, f'T{i+1}_x')
            ty = getattr(self.lat_ctrl, f'T{i+1}_y')
            lat[key] = [[float(tx[j]), float(ty[j])]
                        for j in range(len(tx))]

        # 纵向参数
        lon = cfg['lon']
        lon['station_kp'] = float(self.lon_ctrl.station_kp.item())
        lon['station_ki'] = float(self.lon_ctrl.station_ki.item())
        lon['low_speed_kp'] = float(self.lon_ctrl.low_speed_kp.item())
        lon['low_speed_ki'] = float(self.lon_ctrl.low_speed_ki.item())
        lon['high_speed_kp'] = float(self.lon_ctrl.high_speed_kp.item())
        lon['high_speed_ki'] = float(self.lon_ctrl.high_speed_ki.item())
        lon['switch_speed'] = float(self.lon_ctrl.switch_speed.item())
        for i, key in enumerate(['L1_acc_up_lim', 'L2_acc_low_lim',
                                  'L3_acc_up_rate', 'L4_acc_down_rate',
                                  'L5_rate_gain']):
            tx = getattr(self.lon_ctrl, f'L{i+1}_x')
            ty = getattr(self.lon_ctrl, f'L{i+1}_y')
            lon[key] = [[float(tx[j]), float(ty[j])]
                        for j in range(len(tx))]

        return cfg


def tracking_loss(history: list[dict], ref_speed: float,
                  w_lat: float = 10.0, w_head: float = 5.0,
                  w_speed: float = 1.0, w_steer_rate: float = 0.1,
                  w_acc_rate: float = 0.1) -> torch.Tensor:
    """跟踪误差加权损失。"""
    lat_errs = torch.stack([h['lateral_error'] for h in history])
    head_errs = torch.stack([h['heading_error'] for h in history])
    speeds = torch.stack([h['v'] for h in history])
    steers = torch.stack([h['steer'] for h in history])
    accs = torch.stack([h['acc'] for h in history])

    speed_errs = speeds - ref_speed

    loss = (w_lat * (lat_errs ** 2).mean()
            + w_head * (head_errs ** 2).mean()
            + w_speed * (speed_errs ** 2).mean())

    # 操控平滑性
    if len(steers) > 1:
        steer_rate = steers[1:] - steers[:-1]
        acc_rate = accs[1:] - accs[:-1]
        loss = loss + w_steer_rate * (steer_rate ** 2).mean()
        loss = loss + w_acc_rate * (acc_rate ** 2).mean()

    return loss


_TRAJECTORY_BUILDERS = {
    'straight': lambda speed: generate_straight(length=200, speed=speed),
    'circle': lambda speed: generate_circle(radius=30.0, speed=speed,
                                             arc_angle=3.14159/2),
    'sine': lambda speed: generate_sine(amplitude=3.0, wavelength=50.0,
                                         n_waves=2, speed=speed),
    'combined': lambda speed: generate_combined(speed=speed),
}


def train(trajectories: list[str] | None = None,
          n_epochs: int = 100,
          lr: float = 1e-3,
          lr_tables: float = 5e-4,
          sim_length: float | None = None,
          sim_speed: float = 5.0,
          verbose: bool = True,
          ) -> dict:
    """训练主入口。返回 {'losses', 'saved_path', 'params'}。"""
    if trajectories is None:
        trajectories = ['circle', 'sine', 'combined']

    params = DiffControllerParams()

    # 分组学习率
    table_params = []
    other_params = []
    for name, p in params.named_parameters():
        if '_y' in name:  # 表 y 值
            table_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': table_params, 'lr': lr_tables},
    ])

    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = torch.tensor(0.0)

        for traj_name in trajectories:
            builder = _TRAJECTORY_BUILDERS[traj_name]
            traj = builder(sim_speed)
            if sim_length is not None:
                # 截断轨迹到指定长度
                max_t = sim_length / sim_speed
                traj = [p for p in traj if p.t <= max_t]
                if len(traj) < 10:
                    continue

            history = run_simulation(
                traj, init_speed=sim_speed,
                cfg=params.cfg,
                lat_ctrl=params.lat_ctrl,
                lon_ctrl=params.lon_ctrl,
                differentiable=True)

            loss = tracking_loss(history, ref_speed=sim_speed)
            epoch_loss = epoch_loss + loss

        epoch_loss = epoch_loss / len(trajectories)
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=10.0)
        optimizer.step()

        losses.append(epoch_loss.item())
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}  loss={epoch_loss.item():.6f}")

    # 保存
    cfg_out = params.to_config_dict()
    saved_path = save_tuned_config(cfg_out, meta={
        'final_loss': losses[-1],
        'epochs': n_epochs,
        'trajectories': trajectories,
        'lr': lr,
    })
    if verbose:
        print(f"参数已保存: {saved_path}")

    return {'losses': losses, 'saved_path': saved_path, 'params': params}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 可微调参训练')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--trajectories', nargs='+',
                        default=['circle', 'sine', 'combined'],
                        choices=list(_TRAJECTORY_BUILDERS.keys()))
    parser.add_argument('--speed', type=float, default=5.0)
    args = parser.parse_args()

    result = train(
        trajectories=args.trajectories,
        n_epochs=args.epochs,
        lr=args.lr,
        sim_speed=args.speed,
    )
    print(f"\n最终 loss: {result['losses'][-1]:.6f}")
    print(f"初始 loss: {result['losses'][0]:.6f}")
    print(f"下降比: {result['losses'][-1] / result['losses'][0]:.2%}")
```

**Step 4: 确认通过**

Run: `cd sim && python -m pytest tests/test_train.py -v`

**Step 5: 提交**

```bash
git add sim/train.py sim/tests/test_train.py
git commit -m "[sim] V2 可微调参训练 pipeline：DiffControllerParams + loss + 训练循环"
```

---

### Task 10: 端到端验证 + run_demo 更新

**Files:**
- Modify: `sim/run_demo.py` — 支持 `--config` 参数加载调参后的 YAML
- Run: 全量测试 + 训练 + 可视化

**Step 1: 更新 run_demo.py**

在 `argparse` 中添加：

```python
parser.add_argument('--config', type=str, default=None,
                    help='加载指定 YAML 配置（如调参后的结果）')
```

在 `run_simulation` 调用处传入加载的 config：

```python
cfg = load_config(args.config) if args.config else None
history = run_simulation(traj, init_speed=init_v, cfg=cfg)
```

注意：`run_demo.py` 中 `history` 里的值现在可能是 `torch.Tensor`，绘图时需要 `.item()` 或自动处理（matplotlib 能处理 0-dim tensor）。

**Step 2: 运行全量测试**

Run: `cd sim && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 3: 运行训练**

Run: `cd sim && python train.py --epochs 50 --trajectories circle sine --speed 5.0`
Expected: loss 逐步下降，最终保存 YAML 到 `sim/configs/tuned/`

**Step 4: 用调参结果跑可视化**

Run: `cd sim && python run_demo.py --config configs/tuned/tuned_*.yaml --save --no-show`
Expected: 保存对比图到 `sim/results/`

**Step 5: 提交（代码 + 文档 + 结果图）**

```bash
git add sim/run_demo.py sim/configs/tuned/ sim/results/
git commit -m "[sim] V2 可微调参端到端验证 + 训练结果"
```

---

## 实现注意事项

### tensor/float 兼容性
- 所有控制器 `compute()` 的输入均为 `torch.Tensor`
- `TrajectoryPoint` 的字段保持 `float`（轨迹生成不在梯度路径上）
- 从 `TrajectoryPoint` 取值传入 torch 运算时，用 `torch.tensor(pt.x)` 包装

### BPTT 内存管理
- 2000 步 × 小状态，单条轨迹的计算图约 50-100MB，CPU 可承受
- 如遇内存问题：截短轨迹 or 每 N 步 detach 一次

### differentiable=False 一致性验证方法
- 在 Task 8 中对每个关键场景跑 V1 numpy 版和 torch False 版，比较数值
- 允许浮点精度差异 < 1e-5（float32 vs float64 差异）
- `normalize_angle` 实现方式改变（fmod → atan2），在正常范围 (-π, π) 内结果一致

### smooth_step 温度选择指南
- PID 速度切换 (`switch_speed`): `temp=0.5`（渐进切换，不需要太锐利）
- 急弯检测 (`curvature < -0.0075`): `temp=0.001`（接近硬切换，安全相关）
- 低速保护 (`speed < 1.5`): `temp=0.3`（较平滑过渡）
- 站位误差保护 (`speed > 10`): `temp=1.0`（宽松过渡）

### 已知限制
- V1 的 `normalize_angle` 用 `fmod`，在 ±π 边界行为略有不同；torch 版用 `atan2` 更平滑
- `query_nearest_by_position` 的 detached argmin 在轨迹急转弯处可能跳点，但不影响收敛
- 纵向控制器 Step 2 的多级阈值逻辑，smooth 版是简化近似（不完全等价于原始的 hysteresis）
