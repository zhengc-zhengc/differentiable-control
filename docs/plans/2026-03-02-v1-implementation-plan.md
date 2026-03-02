# V1 可微控制 Pipeline 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 LatControllerTruck + LonController（简化版）+ 运动学自行车模型，50Hz 闭环仿真，在直线/圆弧/正弦/组合轨迹上可视化跟踪效果。

**Architecture:** 纯 Python + numpy 实现（V1 不需要梯度，V2 再转 PyTorch）。控制器严格按照 `docs/controller_spec.md` 的伪代码翻译。运动学自行车模型作为被控对象。仿真主循环 50Hz，横向→纵向→车辆更新。

**Tech Stack:** Python 3.10+, numpy, matplotlib, pyyaml

**配置文件设计:** 所有控制器参数（查找表 T1-T8, L1-L5, PID 增益, 车辆参数等）存放在 `sim/configs/default.yaml` 中，控制器通过 config dict 接收参数，不硬编码。

---

## Task 1: 环境搭建 + common.py 基础组件 + 配置文件

**Files:**
- Create: `sim/common.py`
- Create: `sim/configs/default.yaml`
- Create: `sim/config.py`
- Create: `sim/tests/test_common.py`

**Step 1: 创建虚拟环境并安装依赖**

```bash
cd E:/AI_project/differentiable_control/get_src
uv venv
source .venv/Scripts/activate
uv pip install numpy matplotlib pytest
```

**Step 2: 写 test_common.py 测试**

```python
# sim/tests/test_common.py
"""common.py 基础组件测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common import lookup1d, rate_limit, normalize_angle, PID, TrajectoryPoint


class TestLookup1d:
    def test_exact_match(self):
        table = [(0, 0), (10, 1), (20, 2)]
        assert lookup1d(table, 10) == 1.0

    def test_interpolation(self):
        table = [(0, 0), (10, 1), (20, 2)]
        assert lookup1d(table, 5) == pytest.approx(0.5)

    def test_clamp_low(self):
        table = [(0, 0), (10, 1)]
        assert lookup1d(table, -5) == 0.0

    def test_clamp_high(self):
        table = [(0, 0), (10, 1)]
        assert lookup1d(table, 15) == 1.0

    def test_single_point(self):
        table = [(5, 3)]
        assert lookup1d(table, 0) == 3.0
        assert lookup1d(table, 10) == 3.0


class TestRateLimit:
    def test_within_limit(self):
        # target - prev = 5, max_delta = 10*0.02 = 0.2... no
        # rate=300 deg/s, dt=0.02 => max_delta=6
        result = rate_limit(0, 5, 300, 0.02)
        assert result == pytest.approx(5.0)

    def test_rate_clamped_up(self):
        # rate=120 deg/s, dt=0.02 => max_delta=2.4
        result = rate_limit(0, 10, 120, 0.02)
        assert result == pytest.approx(2.4)

    def test_rate_clamped_down(self):
        result = rate_limit(10, 0, 120, 0.02)
        assert result == pytest.approx(10 - 2.4)


class TestNormalizeAngle:
    def test_zero(self):
        assert normalize_angle(0) == pytest.approx(0)

    def test_pi(self):
        assert abs(normalize_angle(math.pi)) == pytest.approx(math.pi)

    def test_wrap_positive(self):
        assert normalize_angle(3 * math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_wrap_negative(self):
        assert normalize_angle(-3 * math.pi) == pytest.approx(math.pi, abs=1e-10)

    def test_just_over_pi(self):
        assert normalize_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)


class TestPID:
    def test_proportional_only(self):
        pid = PID(kp=1.0, ki=0.0, kd=0.0, integrator_enable=False,
                  integrator_saturation=1.0)
        assert pid.control(2.0, 0.02) == pytest.approx(2.0)

    def test_integral_accumulation(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, integrator_enable=True,
                  integrator_saturation=10.0)
        pid.control(1.0, 0.02)  # integral = 0.02
        pid.control(1.0, 0.02)  # integral = 0.04
        assert pid.control(1.0, 0.02) == pytest.approx(0.06)

    def test_integral_saturation(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, integrator_enable=True,
                  integrator_saturation=0.01)
        for _ in range(100):
            pid.control(1.0, 0.02)
        result = pid.control(1.0, 0.02)
        assert result == pytest.approx(0.01)

    def test_reset(self):
        pid = PID(kp=1.0, ki=1.0, kd=0.0, integrator_enable=True,
                  integrator_saturation=10.0)
        pid.control(1.0, 0.02)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0


class TestTrajectoryPoint:
    def test_fields(self):
        pt = TrajectoryPoint(x=1.0, y=2.0, theta=0.1, kappa=0.01,
                             v=5.0, a=0.0, s=10.0, t=0.0)
        assert pt.x == 1.0
        assert pt.v == 5.0
```

**Step 3: 运行测试确认失败**

```bash
cd E:/AI_project/differentiable_control/get_src
python -m pytest sim/tests/test_common.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'common'`

**Step 4: 实现 common.py**

```python
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
    return table[-1][1]  # fallback


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
            self.integral = clamp(self.integral + error * dt,
                                  -self.sat, self.sat)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def set_pid(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd


class IIR:
    """一阶 IIR 低通滤波器。按 spec §1.3（传递函数形式）。
    y = x - alpha * y_prev
    """

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
    theta: float  # rad, 切线航向
    kappa: float  # 1/m, 曲率（左转正）
    v: float      # m/s, 参考速度
    a: float      # m/s², 参考加速度
    s: float      # m, 累积弧长
    t: float      # s, 相对轨迹起点的时间（relative_time）
```

**Step 5: 运行测试确认通过**

```bash
python -m pytest sim/tests/test_common.py -v
```

Expected: 全部 PASS

**Step 6: 创建 configs/default.yaml 配置文件**

```yaml
# sim/configs/default.yaml
# 可微控制器参数配置 — 所有可调参数集中管理

vehicle:
  wheelbase: 3.5          # m, 轴距
  steer_ratio: 17.5       # 转向传动比
  # 以下参数 V1 简化版暂不使用（CalFinalTorque 需要）
  mass: 2440.0            # kg
  wheel_rolling_radius: 0.505  # m
  windward_area: 8.0      # m²
  transmission_efficiency: 0.85
  transmission_ratio_D: 14.02

simulation:
  dt: 0.02                # s, 控制周期 50Hz

lat_truck:
  # 硬编码常量
  kLh: 0.0                # m, 铰接轴距（未启用）
  rate_limit_fb: 120.0    # °/s
  rate_limit_ff: 165.0    # °/s
  rate_limit_total: 300.0 # °/s
  min_prev_dist: 5.0      # m
  min_reach_dis: 3.0      # m
  min_speed_prot: 0.1     # km/h

  # 查找表 T1-T8，每张表格式: [[index, value], ...]
  T1_max_theta_deg:
    - [0, 3.86]
    - [10, 3.86]
    - [20, 3.86]
    - [30, 3.86]
    - [40, 3.86]
    - [50, 3.86]
    - [60, 3.86]

  T2_prev_time_dist:
    - [0, 1.5]
    - [10, 1.5]
    - [20, 1.5]
    - [30, 1.5]
    - [40, 1.5]
    - [50, 1.5]
    - [60, 1.5]

  T3_reach_time_theta:
    - [0, 1.1]
    - [10, 1.1]
    - [20, 1.1]
    - [30, 1.1]
    - [40, 1.1]
    - [50, 1.1]
    - [60, 1.1]

  T4_T_dt:
    - [0, 0.0]
    - [10, 0.0]
    - [20, 0.3]
    - [30, 0.3]
    - [40, 0.3]
    - [50, 0.3]
    - [60, 0.3]

  T5_near_point_time:
    - [0, 0.1]
    - [10, 0.1]
    - [20, 0.1]
    - [30, 0.1]
    - [40, 0.1]
    - [50, 0.1]
    - [60, 0.1]

  T6_far_point_time:
    - [0, 1.0]
    - [10, 1.0]
    - [20, 1.0]
    - [30, 1.0]
    - [40, 1.0]
    - [50, 1.0]
    - [60, 1.0]

  T7_max_steer_angle:
    - [0, 1100]
    - [10, 1100]
    - [20, 1100]
    - [30, 500]
    - [40, 500]
    - [50, 500]
    - [60, 500]

  T8_slip_param:
    - [0, 1.0]
    - [10, 1.0]
    - [20, 1.0]
    - [30, 1.0]
    - [40, 1.0]
    - [50, 1.0]
    - [60, 1.0]
    - [70, 1.0]
    - [120, 1.0]

lon:
  # PID 参数
  station_kp: 0.25
  station_ki: 0.0
  station_integrator_enable: false
  station_sat: 0.3

  low_speed_kp: 0.35
  low_speed_ki: 0.01
  high_speed_kp: 0.34
  high_speed_ki: 0.01
  speed_pid_sat: 0.3

  switch_speed: 3.0       # m/s

  # 前馈与预览
  preview_window: 5.0           # 拍 (×dt=0.1s)
  preview_window_speed: 50.0    # 拍 (×dt=1.0s)
  acc_use_preview_a: 1.0
  station_error_limit: 8.0      # m
  speed_input_limit: 5.0        # m/s
  acc_standstill_down_rate: -0.005  # m/s²/step
  iir_alpha: 0.15

  # 查找表 L1-L5
  L1_acc_up_lim:
    - [0, 1.6]
    - [10, 1.5]
    - [20, 1.4]
    - [30, 1.3]
    - [40, 1.2]

  L2_acc_low_lim:
    - [0, -0.1]
    - [1, -0.5]
    - [2, -1.5]
    - [4, -2.0]
    - [12, -3.0]
    - [25, -3.5]

  L3_acc_up_rate:
    - [-0.5, 0.045]
    - [-0.25, 0.040]
    - [-0.1, 0.035]
    - [0.1, 0.035]
    - [0.25, 0.040]
    - [0.5, 0.045]

  L4_acc_down_rate:
    - [-1.0, -0.030]
    - [-0.5, -0.030]
    - [-0.3, -0.025]
    - [-0.2, -0.020]
    - [-0.1, -0.020]
    - [0.1, -0.020]
    - [0.2, -0.025]
    - [0.5, -0.030]
    - [1.0, -0.030]

  L5_rate_gain:
    - [0, 1.5]
    - [10, 1.5]
    - [20, 1.35]
    - [30, 1.2]
    - [50, 1.0]
```

**Step 7: 创建 config.py 加载器**

```python
# sim/config.py
"""配置文件加载器。"""
import os
import yaml


def load_config(path: str | None = None) -> dict:
    """加载 YAML 配置文件。默认加载 configs/default.yaml。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__),
                            'configs', 'default.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def table_from_config(entries: list[list[float]]) -> list[tuple[float, float]]:
    """将 YAML 格式的表 [[idx, val], ...] 转为 lookup1d 需要的 tuple 列表。"""
    return [(row[0], row[1]) for row in entries]
```

**Step 8: 提交**

```bash
git add sim/common.py sim/configs/default.yaml sim/config.py sim/tests/test_common.py
git commit -m "[sim] 实现公共基础组件 + YAML 配置文件管理所有控制器参数"
```

---

## Task 2: trajectory.py — 轨迹生成 + TrajectoryAnalyzer

**Files:**
- Create: `sim/trajectory.py`
- Create: `sim/tests/test_trajectory.py`

**Step 1: 写测试**

```python
# sim/tests/test_trajectory.py
"""轨迹生成与分析器测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trajectory import (generate_straight, generate_circle, generate_sine,
                         generate_combined, TrajectoryAnalyzer)


class TestStraightTrajectory:
    def test_length_and_speed(self):
        pts = generate_straight(length=100, speed=10.0, dt=0.02)
        assert len(pts) > 0
        assert pts[0].v == pytest.approx(10.0)
        # 100m at 10m/s = 10s = 500 steps
        assert len(pts) == pytest.approx(500, abs=5)

    def test_heading_zero(self):
        pts = generate_straight(length=50, speed=5.0)
        for p in pts:
            assert p.theta == pytest.approx(0.0, abs=1e-6)
            assert p.kappa == pytest.approx(0.0, abs=1e-6)


class TestCircleTrajectory:
    def test_constant_curvature(self):
        R = 30.0
        pts = generate_circle(radius=R, speed=5.0, arc_angle=math.pi)
        for p in pts:
            assert p.kappa == pytest.approx(1.0 / R, abs=1e-6)

    def test_arc_length(self):
        R = 30.0
        pts = generate_circle(radius=R, speed=5.0, arc_angle=math.pi)
        total_s = pts[-1].s
        expected = R * math.pi
        assert total_s == pytest.approx(expected, rel=0.02)


class TestSineTrajectory:
    def test_returns_points(self):
        pts = generate_sine(amplitude=5.0, wavelength=50.0,
                            n_waves=2, speed=5.0)
        assert len(pts) > 0
        # 起点应该在原点附近
        assert pts[0].x == pytest.approx(0, abs=0.1)
        assert pts[0].y == pytest.approx(0, abs=0.1)


class TestAnalyzer:
    def test_nearest_by_position(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_position(50.0, 0.0)
        assert pt.x == pytest.approx(50.0, abs=0.5)

    def test_nearest_by_relative_time(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_relative_time(1.0)  # 1s at 10m/s = 10m
        assert pt.x == pytest.approx(10.0, abs=0.5)
        assert pt.v == pytest.approx(10.0)

    def test_time_clamp_at_end(self):
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        pt = analyzer.query_nearest_by_relative_time(999.0)
        assert pt.x == pytest.approx(pts[-1].x, abs=0.5)

    def test_frenet_on_track(self):
        """车辆在轨迹上时，横向偏差应为 0。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(50.0, 0.0)
        s, s_dot, d, d_dot = analyzer.to_frenet(50.0, 0.0, 0.0, 10.0, matched)
        assert d == pytest.approx(0.0, abs=0.1)
        assert s_dot == pytest.approx(10.0, abs=0.5)

    def test_frenet_lateral_offset(self):
        """车辆偏左 2m 时，d 应为正（spec 定义左正）。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        matched = analyzer.query_nearest_by_position(50.0, 2.0)
        s, s_dot, d, d_dot = analyzer.to_frenet(50.0, 2.0, 0.0, 10.0, matched)
        assert d == pytest.approx(2.0, abs=0.2)
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest sim/tests/test_trajectory.py -v
```

**Step 3: 实现 trajectory.py**

```python
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
        pts.append(TrajectoryPoint(
            x=x, y=y, theta=heading, kappa=0.0,
            v=speed, a=0.0, s=s, t=t))
        s += speed * dt
    return pts


def generate_circle(radius: float, speed: float,
                    arc_angle: float = 2 * math.pi,
                    dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成圆弧轨迹。从 (R, 0) 开始，圆心在原点，逆时针。
    为了和直线衔接方便，改为从 (0,0) 开始，初始航向 heading=0。
    """
    kappa = 1.0 / radius
    arc_length = radius * arc_angle
    n_steps = int(arc_length / (speed * dt))
    pts = []
    # 圆心在 (0, R)，起点 (0, 0)，初始航向 0（向右）
    cx, cy = 0.0, radius
    s = 0.0
    for i in range(n_steps + 1):
        t = i * dt
        # 行驶弧长对应的角度
        angle = speed * t / radius  # 从 -pi/2 开始的角位移
        x = cx + radius * math.sin(angle)
        y = cy - radius * math.cos(angle)
        theta = angle  # 切线方向
        pts.append(TrajectoryPoint(
            x=x, y=y, theta=theta, kappa=kappa,
            v=speed, a=0.0, s=s, t=t))
        s += speed * dt
    return pts


def generate_sine(amplitude: float, wavelength: float, n_waves: float,
                  speed: float, dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成正弦曲线轨迹。y = A * sin(2*pi*x / lambda)。
    沿 x 轴以近似匀速行驶，用微分几何计算 theta 和 kappa。
    """
    total_x = wavelength * n_waves
    # 用小步离散化 x，然后按弧长重采样
    n_fine = int(total_x / 0.01)  # 0.01m 精度
    xs, ys = [], []
    for i in range(n_fine + 1):
        xi = total_x * i / n_fine
        yi = amplitude * math.sin(2 * math.pi * xi / wavelength)
        xs.append(xi)
        ys.append(yi)

    # 计算弧长和切线
    arc_s = [0.0]
    for i in range(1, len(xs)):
        ds = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
        arc_s.append(arc_s[-1] + ds)
    total_arc = arc_s[-1]

    # 按 speed * dt 等弧长采样
    n_steps = int(total_arc / (speed * dt))
    pts = []
    fine_idx = 0
    for step in range(n_steps + 1):
        target_s = step * speed * dt
        # 找到 fine 中对应的位置
        while fine_idx < len(arc_s) - 2 and arc_s[fine_idx + 1] < target_s:
            fine_idx += 1
        # 在 fine_idx 和 fine_idx+1 之间插值
        if fine_idx >= len(arc_s) - 1:
            fine_idx = len(arc_s) - 2
        frac = 0.0
        ds_seg = arc_s[fine_idx + 1] - arc_s[fine_idx]
        if ds_seg > 1e-9:
            frac = (target_s - arc_s[fine_idx]) / ds_seg
        x = xs[fine_idx] + frac * (xs[fine_idx + 1] - xs[fine_idx])
        y = ys[fine_idx] + frac * (ys[fine_idx + 1] - ys[fine_idx])

        # 解析导数计算 theta 和 kappa
        k = 2 * math.pi / wavelength
        dydx = amplitude * k * math.cos(k * x)
        d2ydx2 = -amplitude * k * k * math.sin(k * x)
        theta = math.atan2(dydx, 1.0)
        kappa = d2ydx2 / (1 + dydx ** 2) ** 1.5

        pts.append(TrajectoryPoint(
            x=x, y=y, theta=theta, kappa=kappa,
            v=speed, a=0.0, s=target_s, t=step * dt))
    return pts


def generate_combined(speed: float, dt: float = 0.02) -> list[TrajectoryPoint]:
    """生成组合轨迹：直线(30m) → 左转圆弧(R=30m, 90°) → 直线(30m)。"""
    pts = []
    s = 0.0
    t = 0.0

    # 段1：直线 30m，heading=0
    seg1_len = 30.0
    n1 = int(seg1_len / (speed * dt))
    for i in range(n1):
        x = speed * i * dt
        pts.append(TrajectoryPoint(x=x, y=0.0, theta=0.0, kappa=0.0,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    # 段2：左转圆弧 R=30m, 90度
    R = 30.0
    arc = math.pi / 2
    arc_len = R * arc
    n2 = int(arc_len / (speed * dt))
    cx = pts[-1].x  # 圆弧起点
    cy = pts[-1].y + R  # 圆心在上方
    start_angle = -math.pi / 2  # 从圆心看，起点在正下方
    for i in range(1, n2 + 1):
        angle = start_angle + (speed * i * dt) / R
        x = cx + R * math.cos(angle)
        y = cy + R * math.sin(angle)
        theta = angle + math.pi / 2  # 切线 = 法线 + 90°
        pts.append(TrajectoryPoint(x=x, y=y, theta=theta, kappa=1.0/R,
                                   v=speed, a=0.0, s=s, t=t))
        s += speed * dt
        t += dt

    # 段3：直线 30m，heading = pi/2（向上）
    seg3_len = 30.0
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
        """返回距离 (x,y) 最近的轨迹点。"""
        best_dist = float('inf')
        best_pt = self.points[0]
        for pt in self.points:
            dist = (pt.x - x) ** 2 + (pt.y - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pt = pt
        return best_pt

    def query_nearest_by_relative_time(self, t_rel: float) -> TrajectoryPoint:
        """按 relative_time 查询，超界夹紧。在相邻点间线性插值。"""
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
        """Frenet 变换。返回 (s_matched, s_dot, d, d_dot)。
        d 定义：左正（与 spec §1.6 一致）。
        """
        heading_err = normalize_angle(theta_rad - matched.theta)
        s_dot = v_mps * math.cos(heading_err)
        d = (math.cos(matched.theta) * (y - matched.y)
             - math.sin(matched.theta) * (x - matched.x))
        d_dot = v_mps * math.sin(heading_err)
        # s_matched: 匹配点弧长 + 沿轨迹投影
        dx_v = x - matched.x
        dy_v = y - matched.y
        proj = dx_v * math.cos(matched.theta) + dy_v * math.sin(matched.theta)
        s_matched = matched.s + proj
        return s_matched, s_dot, d, d_dot
```

**Step 4: 运行测试**

```bash
python -m pytest sim/tests/test_trajectory.py -v
```

Expected: 全部 PASS

**Step 5: 提交**

```bash
git add sim/trajectory.py sim/tests/test_trajectory.py
git commit -m "[sim] 实现轨迹生成（直线/圆弧/正弦/组合）和 TrajectoryAnalyzer"
```

---

## Task 3: vehicle.py — 运动学自行车模型

**Files:**
- Create: `sim/vehicle.py`
- Create: `sim/tests/test_vehicle.py`

**Step 1: 写测试**

```python
# sim/tests/test_vehicle.py
"""运动学自行车模型测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vehicle import BicycleModel


class TestBicycleModel:
    def test_straight_line(self):
        """零转角 + 匀速 → 直线行驶。"""
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=10.0)
        for _ in range(500):  # 10s
            car.step(delta=0.0, acc=0.0)
        assert car.x == pytest.approx(100.0, abs=1.0)
        assert car.y == pytest.approx(0.0, abs=0.01)
        assert car.yaw == pytest.approx(0.0, abs=0.01)

    def test_acceleration(self):
        """从静止加速。"""
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=0.0)
        for _ in range(50):  # 1s at 1 m/s²
            car.step(delta=0.0, acc=1.0)
        assert car.v == pytest.approx(1.0, abs=0.05)
        assert car.x > 0

    def test_circular_motion(self):
        """恒定转角 → 圆周运动。"""
        L = 3.5
        R = 30.0
        delta = math.atan(L / R)  # 前轮转角
        speed = 5.0
        car = BicycleModel(wheelbase=L, x=0, y=0, yaw=0, v=speed)
        # 转一圈的时间 = 2*pi*R / speed
        n_steps = int(2 * math.pi * R / speed / 0.02)
        for _ in range(n_steps):
            car.step(delta=delta, acc=0.0)
        # 应回到起点附近
        assert car.x == pytest.approx(0.0, abs=2.0)
        assert car.y == pytest.approx(0.0, abs=2.0)

    def test_speed_non_negative(self):
        """速度不能为负。"""
        car = BicycleModel(wheelbase=3.5, x=0, y=0, yaw=0, v=1.0)
        for _ in range(200):  # 强制减速
            car.step(delta=0.0, acc=-5.0)
        assert car.v >= 0.0
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest sim/tests/test_vehicle.py -v
```

**Step 3: 实现 vehicle.py**

```python
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
        """前进一步。delta = 前轮转角(rad), acc = 加速度(m/s²)。"""
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += self.v * math.tan(delta) / self.L * self.dt
        self.v += acc * self.dt
        self.v = max(0.0, self.v)  # 速度不为负

    @property
    def speed_kph(self) -> float:
        return self.v * 3.6

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)
```

**Step 4: 运行测试**

```bash
python -m pytest sim/tests/test_vehicle.py -v
```

Expected: 全部 PASS

**Step 5: 提交**

```bash
git add sim/vehicle.py sim/tests/test_vehicle.py
git commit -m "[sim] 实现运动学自行车模型"
```

---

## Task 4: controller/lat_truck.py — 重卡横向控制器

**Files:**
- Create: `sim/controller/__init__.py`
- Create: `sim/controller/lat_truck.py`
- Create: `sim/tests/test_lat_truck.py`

**Step 1: 写测试**

```python
# sim/tests/test_lat_truck.py
"""LatControllerTruck 测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lat_truck import LatControllerTruck
from trajectory import generate_straight, generate_circle, TrajectoryAnalyzer

CFG = load_config()


class TestLatTruckBasic:
    def test_on_straight_no_steer(self):
        """车辆在直线上、无偏差 → 转向角应接近 0。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _ = ctrl.compute(
            x=50.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer) < 5.0  # 应接近零

    def test_lateral_offset_corrects(self):
        """车辆在直线左偏 2m → 转向角应向右修正（负值）。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _ = ctrl.compute(
            x=50.0, y=2.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        # 左偏 → lateral_error > 0 → dis2lane < 0 → 应向右转（负转角）
        assert steer < 0

    def test_circle_has_steer(self):
        """圆弧轨迹上 → 应有前馈转向角。"""
        R = 30.0
        pts = generate_circle(radius=R, speed=5.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        # 车辆在圆弧起点，航向正确
        steer, _, _, kfar = ctrl.compute(
            x=0.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            yawrate=5.0/R, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True)
        assert abs(steer) > 1.0  # 弯道应有明显转向

    def test_disable_returns_feedback(self):
        """ctrl_enable=False → 输出 = steer_feedback。"""
        pts = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LatControllerTruck(CFG)
        steer, _, _, _ = ctrl.compute(
            x=0.0, y=5.0, yaw_deg=10.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=42.0,
            analyzer=analyzer, ctrl_enable=False)
        assert steer == pytest.approx(42.0)
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest sim/tests/test_lat_truck.py -v
```

**Step 3: 实现 lat_truck.py**

```python
# sim/controller/__init__.py
```

```python
# sim/controller/lat_truck.py
"""重卡横向控制器 — 按 controller_spec.md §2.5 实现。参数从配置文件加载。"""
import math
from common import lookup1d, rate_limit, clamp, sign, normalize_angle
from config import table_from_config
from trajectory import TrajectoryAnalyzer

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class LatControllerTruck:
    """重卡横向控制器。所有参数从 config dict 加载。"""

    def __init__(self, cfg: dict):
        veh = cfg['vehicle']
        lat = cfg['lat_truck']

        self.wheelbase = veh['wheelbase']
        self.steer_ratio = veh['steer_ratio']

        # 常量参数
        self.kLh = lat['kLh']
        self.rate_limit_fb = lat['rate_limit_fb']
        self.rate_limit_ff = lat['rate_limit_ff']
        self.rate_limit_total = lat['rate_limit_total']
        self.min_prev_dist = lat['min_prev_dist']
        self.min_reach_dis = lat['min_reach_dis']
        self.min_speed_prot = lat['min_speed_prot']

        # 查找表
        self.T1 = table_from_config(lat['T1_max_theta_deg'])
        self.T2 = table_from_config(lat['T2_prev_time_dist'])
        self.T3 = table_from_config(lat['T3_reach_time_theta'])
        self.T4 = table_from_config(lat['T4_T_dt'])
        self.T5 = table_from_config(lat['T5_near_point_time'])
        self.T6 = table_from_config(lat['T6_far_point_time'])
        self.T7 = table_from_config(lat['T7_max_steer_angle'])
        self.T8 = table_from_config(lat['T8_slip_param'])

        # 内部状态
        self.steer_fb_prev = 0.0
        self.steer_ff_prev = 0.0
        self.steer_total_prev = 0.0

    def compute(self, x: float, y: float, yaw_deg: float,
                speed_kph: float, yawrate: float, steer_feedback: float,
                analyzer: TrajectoryAnalyzer, ctrl_enable: bool,
                dt: float = 0.02
                ) -> tuple[float, float, float, float]:
        """计算转向角。返回 (steering_target, kappa_current, kappa_near, kappa_far)。"""
        speed_kph = max(speed_kph, self.min_speed_prot)
        speed_mps = speed_kph / 3.6
        yaw_rad = yaw_deg * DEG2RAD

        if not ctrl_enable:
            self.steer_fb_prev = 0.0
            self.steer_ff_prev = 0.0
            self.steer_total_prev = steer_feedback
            return steer_feedback, 0.0, 0.0, 0.0

        # Step 1: 查表
        max_theta_deg = lookup1d(self.T1, speed_kph)
        prev_time_dist = lookup1d(self.T2, speed_kph)
        reach_time_theta = lookup1d(self.T3, speed_kph)
        T_dt = lookup1d(self.T4, speed_kph)
        near_pt_time = lookup1d(self.T5, speed_kph)
        far_pt_time = lookup1d(self.T6, speed_kph)
        max_steer_angle = lookup1d(self.T7, speed_kph)
        slip_param = lookup1d(self.T8, speed_kph)

        # Step 2: 轨迹查询
        currt = analyzer.query_nearest_by_position(x, y)
        near = analyzer.query_nearest_by_relative_time(
            currt.t + near_pt_time)
        far = analyzer.query_nearest_by_relative_time(
            currt.t + far_pt_time)

        # Step 3: 误差计算
        dx = x - currt.x
        dy = y - currt.y
        lateral_error = math.cos(currt.theta) * dy - math.sin(currt.theta) * dx
        heading_error = normalize_angle(yaw_rad - currt.theta)
        curvature_far = far.kappa

        # Step 4: real_theta (kLh=0 → real_theta = -heading_error)
        vehicle_speed_clamped = clamp(speed_mps, 1.0, 100.0)
        real_theta = -heading_error - math.atan(
            self.kLh * yawrate / vehicle_speed_clamped)

        # Step 5: real_dt_theta
        real_dt_theta = -(yawrate - curvature_far * speed_mps)

        # Step 6: target_theta
        prev_dist = max(speed_mps * prev_time_dist, self.min_prev_dist)
        dis2lane = -lateral_error
        error_angle_raw = math.atan(dis2lane / prev_dist)
        max_err_angle = min(max_theta_deg * DEG2RAD, abs(error_angle_raw))
        target_theta = sign(error_angle_raw) * max_err_angle

        target_dt_theta = (math.sin(real_theta) * speed_mps * prev_dist
                           / (prev_dist ** 2 + dis2lane ** 2) * -1.0)

        # Step 7: target_curvature
        denom = max(reach_time_theta * speed_mps, self.min_reach_dis)
        target_curvature = -((target_theta - real_theta)
                             + (target_dt_theta - real_dt_theta) * T_dt) / denom

        # Step 8: 反馈转向角
        steer_fb_raw = (math.atan(target_curvature * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_fb = rate_limit(self.steer_fb_prev, steer_fb_raw,
                              self.rate_limit_fb, dt)
        self.steer_fb_prev = steer_fb

        # Step 9: 前馈转向角
        steer_ff_raw = (math.atan(curvature_far * self.wheelbase)
                        * RAD2DEG * self.steer_ratio * slip_param)
        steer_ff = rate_limit(self.steer_ff_prev, steer_ff_raw,
                              self.rate_limit_ff, dt)
        self.steer_ff_prev = steer_ff

        # Step 10: 合并输出
        steer_raw = clamp(steer_fb + steer_ff,
                          -max_steer_angle, max_steer_angle)
        steer_out = rate_limit(self.steer_total_prev, steer_raw,
                               self.rate_limit_total, dt)
        self.steer_total_prev = steer_out

        return steer_out, currt.kappa, near.kappa, curvature_far
```

**Step 4: 运行测试**

```bash
python -m pytest sim/tests/test_lat_truck.py -v
```

Expected: 全部 PASS

**Step 5: 提交**

```bash
git add sim/controller/__init__.py sim/controller/lat_truck.py sim/tests/test_lat_truck.py
git commit -m "[sim] 实现 LatControllerTruck（spec §2.5 完整 10 步）"
```

---

## Task 5: controller/lon.py — 纵向控制器（简化版）

**Files:**
- Create: `sim/controller/lon.py`
- Create: `sim/tests/test_lon.py`

**Step 1: 写测试**

```python
# sim/tests/test_lon.py
"""LonController 简化版测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lon import LonController
from trajectory import generate_straight, TrajectoryAnalyzer

CFG = load_config()


class TestLonBasic:
    def test_on_track_no_correction(self):
        """车辆在轨迹上、速度匹配 → 加速度应接近 0。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert abs(acc) < 0.5

    def test_too_slow_accelerates(self):
        """车速比参考慢 → 应输出正加速度。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert acc > 0

    def test_too_fast_decelerates(self):
        """车速比参考快 → 应输出负加速度。"""
        pts = generate_straight(length=200, speed=5.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=54.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=False)
        assert acc < 0

    def test_acc_within_limits(self):
        """加速度应在 L1/L2 限幅范围内。"""
        pts = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(pts)
        ctrl = LonController(CFG)
        for _ in range(100):
            acc = ctrl.compute(
                x=10.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
                accel_mps2=0.0, curvature_far=0.0,
                analyzer=analyzer, t_now=1.0,
                ctrl_enable=True, ctrl_first_active=False)
            assert -4.0 <= acc <= 2.0
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest sim/tests/test_lon.py -v
```

**Step 3: 实现 lon.py**

```python
# sim/controller/lon.py
"""纵向控制器简化版 — 按 controller_spec.md §4.5 Steps 1-6 实现。
跳过 Steps 7-9（GearControl, CalFinalTorque）。
直接输出加速度指令 (m/s²)。参数从配置文件加载。
"""
import math
from common import lookup1d, clamp, normalize_angle, PID, IIR
from config import table_from_config
from trajectory import TrajectoryAnalyzer

DEG2RAD = math.pi / 180.0


class LonController:
    """纵向控制器（简化版：输出加速度，跳过 Gear/Torque）。
    所有参数从 config dict 加载。
    """

    def __init__(self, cfg: dict):
        lon = cfg['lon']

        # PID 参数
        self.station_kp = lon['station_kp']
        self.station_ki = lon['station_ki']
        self.low_speed_kp = lon['low_speed_kp']
        self.low_speed_ki = lon['low_speed_ki']
        self.high_speed_kp = lon['high_speed_kp']
        self.high_speed_ki = lon['high_speed_ki']
        self.switch_speed = lon['switch_speed']

        # 其他参数
        self.preview_window = lon['preview_window']
        self.preview_window_speed = lon['preview_window_speed']
        self.acc_use_preview_a = lon['acc_use_preview_a']
        self.station_error_limit = lon['station_error_limit']
        self.speed_input_limit = lon['speed_input_limit']
        self.acc_standstill_down_rate = lon['acc_standstill_down_rate']

        # 查找表
        self.L1 = table_from_config(lon['L1_acc_up_lim'])
        self.L2 = table_from_config(lon['L2_acc_low_lim'])
        self.L3 = table_from_config(lon['L3_acc_up_rate'])
        self.L4 = table_from_config(lon['L4_acc_down_rate'])
        self.L5 = table_from_config(lon['L5_rate_gain'])

        # PID 控制器
        pid_sat = lon['speed_pid_sat']
        self.station_pid = PID(
            kp=self.station_kp, ki=self.station_ki, kd=0.0,
            integrator_enable=lon['station_integrator_enable'],
            integrator_saturation=lon['station_sat'])
        self.speed_pid = PID(
            kp=self.low_speed_kp, ki=self.low_speed_ki, kd=0.0,
            integrator_enable=True, integrator_saturation=pid_sat)

        # 内部状态
        self.acc_out_prev = 0.0
        self.iir_acc = IIR(alpha=lon['iir_alpha'])
        self.station_error_fnl_prev = 0.0

    def compute(self, x: float, y: float, yaw_deg: float,
                speed_kph: float, accel_mps2: float,
                curvature_far: float,
                analyzer: TrajectoryAnalyzer, t_now: float,
                ctrl_enable: bool, ctrl_first_active: bool,
                dt: float = 0.02) -> float:
        """计算加速度指令 (m/s²)。"""
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

        station_error = ref_pt.s - s_match
        preview_speed_error = spd_pt.v - speed_mps
        preview_accel_ref = prev_pt.a

        # Step 2: 站位误差保护
        station_limited = clamp(station_error,
                                -self.station_error_limit,
                                self.station_error_limit)
        if speed_kph > 10:
            station_fnl = station_limited
        elif station_limited <= 0.25:
            station_fnl = min(0.0, station_limited)
        elif station_limited >= 0.8:
            station_fnl = station_limited
        elif self.station_error_fnl_prev <= 0.01:
            station_fnl = self.station_error_fnl_prev
        else:
            station_fnl = station_limited
        self.station_error_fnl_prev = station_fnl

        # Step 3: 站位 PID
        speed_offset = self.station_pid.control(station_fnl, dt)

        # Step 4: 速度 PID
        if speed_mps <= self.switch_speed:
            self.speed_pid.set_pid(self.low_speed_kp, self.low_speed_ki, 0.0)
        else:
            self.speed_pid.set_pid(self.high_speed_kp, self.high_speed_ki, 0.0)

        speed_input = clamp(speed_offset + preview_speed_error,
                            -self.speed_input_limit, self.speed_input_limit)
        acc_closeloop = self.speed_pid.control(speed_input, dt)

        # Step 5: 前馈叠加
        acc_cmd = acc_closeloop + self.acc_use_preview_a * preview_accel_ref

        # Step 6: CalFinalAccCmd
        if ctrl_enable:
            acc_up_lim = lookup1d(self.L1, abs(speed_kph))
            acc_low_lim = lookup1d(self.L2, abs(speed_kph))
            acc_up_rate_raw = lookup1d(self.L3, self.acc_out_prev)
            acc_dn_rate_raw = lookup1d(self.L4, self.acc_out_prev)
            rate_gain = lookup1d(self.L5, abs(speed_kph))
            acc_up_rate = acc_up_rate_raw * rate_gain

            # 急弯收紧
            if curvature_far < -0.0075:
                acc_up_lim *= 0.75
                acc_low_lim *= 0.60

            # 低速保护
            if abs(speed_mps) < 1.5:
                acc_dn_rate = self.acc_standstill_down_rate
            else:
                acc_dn_rate = acc_dn_rate_raw

            # 幅值截幅
            acc_clamped = clamp(acc_cmd, acc_low_lim, acc_up_lim)

            # 低速额外下限保护
            if abs(speed_mps) >= 0.2 or acc_clamped >= 0.25:
                acc_lowspd = acc_clamped
            else:
                acc_lowspd = min(-0.05, acc_clamped)

            # 速率截幅
            acc_limited = clamp(acc_lowspd,
                                self.acc_out_prev + acc_dn_rate,
                                self.acc_out_prev + acc_up_rate)
            self.acc_out_prev = acc_limited
        else:
            acc_limited = 0.0
            self.acc_out_prev = 0.0

        # IIR 低通
        acc_out = self.iir_acc.update(acc_limited)
        return acc_out
```

**Step 4: 运行测试**

```bash
python -m pytest sim/tests/test_lon.py -v
```

Expected: 全部 PASS

**Step 5: 提交**

```bash
git add sim/controller/lon.py sim/tests/test_lon.py
git commit -m "[sim] 实现 LonController 简化版（spec §4.5 Steps 1-6，输出加速度）"
```

---

## Task 6: sim_loop.py — 闭环仿真主循环

**Files:**
- Create: `sim/sim_loop.py`
- Create: `sim/tests/test_sim_loop.py`

**Step 1: 写测试**

```python
# sim/tests/test_sim_loop.py
"""闭环仿真测试。"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim_loop import run_simulation
from trajectory import generate_straight, generate_circle


class TestSimLoop:
    def test_straight_line_tracks(self):
        """直线轨迹应能跟踪，最终横向误差很小。"""
        traj = generate_straight(length=200, speed=10.0)
        history = run_simulation(traj, init_speed=10.0)
        # 检查最后 2 秒的横向误差
        n_last = 100  # 2s at 50Hz
        for rec in history[-n_last:]:
            assert abs(rec['lateral_error']) < 1.0, \
                f"lateral_error={rec['lateral_error']:.3f} too large"

    def test_history_has_required_fields(self):
        traj = generate_straight(length=100, speed=5.0)
        history = run_simulation(traj, init_speed=5.0)
        assert len(history) > 0
        rec = history[0]
        for key in ['t', 'x', 'y', 'yaw', 'v', 'steer', 'acc',
                     'lateral_error', 'heading_error',
                     'ref_x', 'ref_y']:
            assert key in rec, f"Missing key: {key}"

    def test_circle_tracks(self):
        """圆弧轨迹应能跟踪。"""
        traj = generate_circle(radius=30.0, speed=5.0,
                               arc_angle=math.pi)
        history = run_simulation(traj, init_speed=5.0)
        # 检查后半段横向误差
        n = len(history)
        for rec in history[n // 2:]:
            assert abs(rec['lateral_error']) < 3.0
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest sim/tests/test_sim_loop.py -v
```

**Step 3: 实现 sim_loop.py**

```python
# sim/sim_loop.py
"""50Hz 闭环仿真主循环。参数从配置文件加载。"""
import math
from common import normalize_angle, TrajectoryPoint
from config import load_config
from trajectory import TrajectoryAnalyzer
from vehicle import BicycleModel
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController

DEG2RAD = math.pi / 180.0


def run_simulation(trajectory: list[TrajectoryPoint],
                   init_speed: float = 0.0,
                   init_x: float | None = None,
                   init_y: float | None = None,
                   init_yaw: float | None = None,
                   cfg: dict | None = None,
                   ) -> list[dict]:
    """运行闭环仿真。返回历史记录。"""
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
                       yaw=yaw0, v=init_speed, dt=dt)
    lat_ctrl = LatControllerTruck(cfg)
    lon_ctrl = LonController(cfg)

    history = []
    n_steps = int(traj_duration / dt)
    prev_steer = 0.0

    for step in range(n_steps):
        t = step * dt

        # 当前参考点
        ref_pt = analyzer.query_nearest_by_position(car.x, car.y)
        dx = car.x - ref_pt.x
        dy = car.y - ref_pt.y
        lateral_error = (math.cos(ref_pt.theta) * dy
                         - math.sin(ref_pt.theta) * dx)
        heading_error = normalize_angle(car.yaw - ref_pt.theta)

        # 估算横摆角速度（用上一步的前轮转角）
        delta_prev = prev_steer / steer_ratio * DEG2RAD
        yawrate = car.v * math.tan(delta_prev) / wheelbase

        # 横向控制器
        steer_out, kappa_cur, kappa_near, curvature_far = lat_ctrl.compute(
            x=car.x, y=car.y,
            yaw_deg=math.degrees(car.yaw),
            speed_kph=car.speed_kph,
            yawrate=yawrate,
            steer_feedback=prev_steer,
            analyzer=analyzer,
            ctrl_enable=True, dt=dt)

        # 纵向控制器
        acc_cmd = lon_ctrl.compute(
            x=car.x, y=car.y,
            yaw_deg=math.degrees(car.yaw),
            speed_kph=car.speed_kph,
            accel_mps2=0.0,
            curvature_far=curvature_far,
            analyzer=analyzer, t_now=t,
            ctrl_enable=True,
            ctrl_first_active=(step == 0), dt=dt)

        history.append({
            't': t, 'x': car.x, 'y': car.y, 'yaw': car.yaw,
            'v': car.v, 'steer': steer_out, 'acc': acc_cmd,
            'lateral_error': lateral_error, 'heading_error': heading_error,
            'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
        })

        # 车辆更新
        delta_front = steer_out / steer_ratio * DEG2RAD
        car.step(delta=delta_front, acc=acc_cmd)
        prev_steer = steer_out

    return history
```

**Step 4: 运行测试**

```bash
python -m pytest sim/tests/test_sim_loop.py -v
```

Expected: 全部 PASS

**Step 5: 提交**

```bash
git add sim/sim_loop.py sim/tests/test_sim_loop.py
git commit -m "[sim] 实现 50Hz 闭环仿真主循环"
```

---

## Task 7: run_demo.py — 可视化入口

**Files:**
- Create: `sim/run_demo.py`

**Step 1: 实现 run_demo.py**

```python
# sim/run_demo.py
"""V1 可视化 Demo：4 种轨迹 × 4 张图。"""
import math
import matplotlib.pyplot as plt
from trajectory import (generate_straight, generate_circle,
                        generate_sine, generate_combined)
from sim_loop import run_simulation


def plot_scenario(name: str, history: list[dict], traj_pts):
    """画 4 张子图：轨迹对比、横向误差、速度跟踪、转向角。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(name, fontsize=14)

    ts = [h['t'] for h in history]

    # 1. 轨迹对比
    ax = axes[0, 0]
    ax.plot([p.x for p in traj_pts], [p.y for p in traj_pts],
            'b--', label='reference', linewidth=1)
    ax.plot([h['x'] for h in history], [h['y'] for h in history],
            'r-', label='actual', linewidth=1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('trajectory')
    ax.grid(True)

    # 2. 横向误差
    ax = axes[0, 1]
    ax.plot(ts, [h['lateral_error'] for h in history], 'g-')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('lateral error (m)')
    ax.set_title('lateral error')
    ax.grid(True)

    # 3. 速度跟踪
    ax = axes[1, 0]
    ax.plot(ts, [h['v'] for h in history], 'r-', label='actual')
    # 参考速度
    ref_v = traj_pts[0].v
    ax.axhline(y=ref_v, color='b', linestyle='--', label=f'ref={ref_v:.1f}m/s')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (m/s)')
    ax.set_title('speed tracking')
    ax.legend()
    ax.grid(True)

    # 4. 转向角
    ax = axes[1, 1]
    ax.plot(ts, [h['steer'] for h in history], 'm-')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('steering angle (deg)')
    ax.set_title('steering angle')
    ax.grid(True)

    plt.tight_layout()
    return fig


def main():
    scenarios = {
        'Straight Line (10 m/s)': {
            'traj': generate_straight(length=200, speed=10.0),
            'init_speed': 10.0,
        },
        'Circle Arc (R=30m, 5 m/s)': {
            'traj': generate_circle(radius=30.0, speed=5.0,
                                    arc_angle=math.pi),
            'init_speed': 5.0,
        },
        'Sine Curve (A=3m, λ=50m, 5 m/s)': {
            'traj': generate_sine(amplitude=3.0, wavelength=50.0,
                                  n_waves=2, speed=5.0),
            'init_speed': 5.0,
        },
        'Combined (straight→arc→straight, 5 m/s)': {
            'traj': generate_combined(speed=5.0),
            'init_speed': 5.0,
        },
    }

    figs = []
    for name, cfg in scenarios.items():
        print(f"Running: {name}...")
        history = run_simulation(cfg['traj'], init_speed=cfg['init_speed'])
        fig = plot_scenario(name, history, cfg['traj'])
        figs.append(fig)

    plt.show()


if __name__ == '__main__':
    main()
```

**Step 2: 运行 demo**

```bash
cd E:/AI_project/differentiable_control/get_src/sim
python run_demo.py
```

Expected: 弹出 4 个 matplotlib 窗口，每个有 4 张子图。观察跟踪效果。

**Step 3: 提交**

```bash
git add sim/run_demo.py
git commit -m "[sim] 实现 V1 可视化 Demo：4 种轨迹闭环仿真"
```

---

## Task 8: 删除 .gitkeep 文件 + 最终验证

**Step 1: 删除不再需要的 .gitkeep**

```bash
cd E:/AI_project/differentiable_control/get_src
rm sim/controller/.gitkeep sim/model/.gitkeep sim/optim/.gitkeep sim/tests/.gitkeep
```

**Step 2: 运行全部测试**

```bash
python -m pytest sim/tests/ -v
```

Expected: 全部 PASS

**Step 3: 运行 demo 确认可视化正常**

```bash
cd sim && python run_demo.py
```

**Step 4: 提交**

```bash
cd E:/AI_project/differentiable_control/get_src
git add -A sim/
git commit -m "[sim] V1 完成：清理 .gitkeep，全部测试通过"
```
