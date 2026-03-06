# 双被控对象集成 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 sim/ 下集成动力学车辆模型（VehicleDynamics），与现有运动学模型（BicycleModel）通过配置/命令行自由切换。

**Architecture:** 适配器模式——将 VehicleDynamics 封装为与 BicycleModel 相同的接口（`step(delta, acc)` + `x/y/yaw/v` 属性），通过 `vehicle_factory.create_vehicle()` 统一创建。sim_loop 和控制器代码零改动。

**Tech Stack:** PyTorch, YAML 配置

---

### Task 1: 新建 DynamicVehicle 适配器

**Files:**
- Create: `sim/model/dynamic_vehicle.py`
- Test: `sim/tests/test_dynamic_vehicle.py`

**Step 1: 写测试文件**

```python
# sim/tests/test_dynamic_vehicle.py
"""动力学车辆适配器测试。"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.dynamic_vehicle import DynamicVehicle

# 默认参数（与 plant 项目一致的 B 级车参数）
DEFAULT_PARAMS = {
    'mass': 2440.0,
    'Iz': 9564.8,
    'lf': 1.354,
    'lr': 1.446,
    'wheel_radius': 0.329,
    'drag_coeff': 0.558,
    'frontal_area': 5.903,
    'air_density': 1.225,
    'rolling_coeff': 0.0065,
    'corner_stiff_f': 80000.0,
    'corner_stiff_r': 80000.0,
    'tire_friction_mu': 0.85,
    'track_width': 1.725,
    'steer_ratio': 16.39,
}


class TestDynamicVehicle:
    def test_interface_has_required_attributes(self):
        """适配器必须暴露与 BicycleModel 一致的属性。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=1.0, y=2.0, yaw=0.1, v=5.0, dt=0.02)
        assert isinstance(car.x, torch.Tensor)
        assert isinstance(car.y, torch.Tensor)
        assert isinstance(car.yaw, torch.Tensor)
        assert isinstance(car.v, torch.Tensor)
        assert hasattr(car, 'speed_kph')
        assert hasattr(car, 'yaw_deg')
        assert hasattr(car, 'step')
        assert hasattr(car, 'detach_state')

    def test_straight_line(self):
        """零转角+零加速度，直行应保持 y≈0、yaw≈0。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        for _ in range(500):  # 10s
            car.step(delta=0.0, acc=0.0)
        # 动力学模型有风阻/滚阻会减速，x 应前进但 < 100m
        assert car.x.item() > 50.0
        assert abs(car.y.item()) < 0.5
        assert abs(car.yaw.item()) < 0.01

    def test_acceleration(self):
        """施加正加速度，速度应增大。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=1.0, dt=0.02)
        v0 = car.v.item()
        for _ in range(50):
            car.step(delta=0.0, acc=1.0)
        assert car.v.item() > v0

    def test_steering_causes_lateral_motion(self):
        """施加转向，y 应偏移。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        delta_front = 0.05  # rad，前轮转角
        for _ in range(200):
            car.step(delta=delta_front, acc=0.0)
        assert abs(car.y.item()) > 0.5

    def test_properties(self):
        """speed_kph 和 yaw_deg 属性。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        assert car.speed_kph.item() == pytest.approx(36.0, abs=0.1)
        assert car.yaw_deg.item() == pytest.approx(0.0, abs=0.01)

    def test_detach_state(self):
        """detach_state 应截断梯度链。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        car.step(delta=0.0, acc=torch.tensor(1.0, requires_grad=True))
        car.detach_state()
        assert not car.x.requires_grad
        assert not car.v.requires_grad

    def test_differentiable_gradient_flows(self):
        """differentiable 模式下梯度应能回传到 acc 输入。"""
        car = DynamicVehicle(DEFAULT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        acc = torch.tensor(1.0, requires_grad=True)
        car.step(delta=0.0, acc=acc)
        car.v.backward()
        assert acc.grad is not None
        assert acc.grad.item() != 0.0
```

**Step 2: 运行测试确认失败**

Run: `cd sim && python -m pytest tests/test_dynamic_vehicle.py -v`
Expected: FAIL（模块不存在）

**Step 3: 实现 DynamicVehicle**

```python
# sim/model/dynamic_vehicle.py
"""动力学车辆模型适配器。

将 6-DOF 动力学模型（VehicleDynamics）封装为与 BicycleModel 相同的接口：
- step(delta, acc): delta 是前轮转角(rad)，acc 是加速度(m/s²)
- 属性: x, y, yaw, v, speed_kph, yaw_deg
- detach_state(): 截断梯度链（Truncated BPTT）

内部将 acc 转换为后驱扭矩分配：T_rl = T_rr = (m * acc / 2) * R，T_fl = T_fr = 0。
使用 RK4 积分。
"""
import torch
import torch.nn as nn


class VehicleDynamics(nn.Module):
    """6-DOF 动力学模型核心。

    state: [x_f, y_f, yaw, vx_f, vy_f, r] — 前轴参考点
    control: [delta_sw, T_fl, T_fr, T_rl, T_rr] — 方向盘角 + 四轮扭矩
    """
    def __init__(self, params):
        super().__init__()
        self.m = float(params['mass'])
        self.Iz = float(params['Iz'])
        self.lf = float(params['lf'])
        self.lr = float(params['lr'])
        self.R = float(params['wheel_radius'])
        self.Cd = float(params['drag_coeff'])
        self.Af = float(params['frontal_area'])
        self.rho = float(params['air_density'])
        self.Crr = float(params['rolling_coeff'])
        self.C_alpha_f = float(params['corner_stiff_f'])
        self.C_alpha_r = float(params['corner_stiff_r'])
        self.mu = float(params.get('tire_friction_mu', 0.85))
        self.track_width = float(params.get('track_width', 1.725))
        self.reverse_sign_speed_mps = max(
            float(params.get('reverse_sign_speed_mps', 0.5)), 1.0e-4)
        self.steer_ratio = max(
            float(params.get('steer_ratio', 16.39)), 1.0e-6)
        self.g = 9.81
        self.Fz_front = 0.5164 * self.m * self.g
        self.Fz_rear = 0.4836 * self.m * self.g
        self._speed_eps = 1.0e-8
        self._force_eps = 1.0e-8

    def derivatives(self, state, control):
        """计算状态导数。state: [B,6], control: [B,5] -> derivatives: [B,6]"""
        x_f, y_f, yaw = state[:, 0], state[:, 1], state[:, 2]
        vx_f, vy_f, r = state[:, 3], state[:, 4], state[:, 5]

        delta = control[:, 0] / self.steer_ratio
        torque_fl, torque_fr = control[:, 1], control[:, 2]
        torque_rl, torque_rr = control[:, 3], control[:, 4]

        L = self.lf + self.lr
        vx_r = vx_f
        vy_r = vy_f - r * L
        vx_cg = vx_r
        vy_cg = vy_r + self.lr * r

        vx_mag = torch.sqrt(vx_r * vx_r + self.reverse_sign_speed_mps ** 2)
        alpha_f = torch.atan2(vy_cg + self.lf * r, vx_mag + self._speed_eps) - delta
        alpha_r = torch.atan2(vy_cg - self.lr * r, vx_mag + self._speed_eps)

        travel_dir = torch.tanh(vx_r / self.reverse_sign_speed_mps)
        fy_f0 = -self.C_alpha_f * alpha_f * travel_dir
        fy_r0 = -self.C_alpha_r * alpha_r * travel_dir

        fy_f_max = self.mu * (self.Fz_front * 0.5)
        fy_r_max = self.mu * (self.Fz_rear * 0.5)
        fy_f = fy_f_max * torch.tanh(fy_f0 / (fy_f_max + self._force_eps))
        fy_r = fy_r_max * torch.tanh(fy_r0 / (fy_r_max + self._force_eps))

        fx_fl = torque_fl / self.R
        fx_fr = torque_fr / self.R
        fx_rl = torque_rl / self.R
        fx_rr = torque_rr / self.R

        speed = torch.sqrt(vx_cg * vx_cg + vy_cg * vy_cg + self._speed_eps)
        roll_force = self.Crr * self.m * self.g * torch.tanh(10.0 * vx_cg)
        aero_force = 0.5 * self.rho * self.Cd * self.Af * speed * speed

        cos_delta = torch.cos(delta)
        sin_delta = torch.sin(delta)

        fx_front = (fx_fl + fx_fr) * cos_delta - (2.0 * fy_f) * sin_delta
        fy_front = (fx_fl + fx_fr) * sin_delta + (2.0 * fy_f) * cos_delta
        fx_rear = fx_rl + fx_rr
        fy_rear = 2.0 * fy_r

        fx_total = fx_front + fx_rear - roll_force - aero_force
        fy_total = fy_front + fy_rear

        yaw_moment = (
            fy_front * self.lf
            - fy_rear * self.lr
            + (fx_fr - fx_fl) * (self.track_width * 0.5)
            + (fx_rr - fx_rl) * (self.track_width * 0.5)
        )

        dvx_r = (fx_total + self.m * vy_r * r) / self.m
        dvy_r = (fy_total - self.m * vx_r * r) / self.m
        dr = yaw_moment / self.Iz
        dvx_f = dvx_r
        dvy_f = dvy_r + dr * L

        dx_f = vx_f * torch.cos(yaw) - vy_f * torch.sin(yaw)
        dy_f = vx_f * torch.sin(yaw) + vy_f * torch.cos(yaw)
        dyaw = r

        return torch.stack([dx_f, dy_f, dyaw, dvx_f, dvy_f, dr], dim=1)

    def rk4_step(self, state, control, dt):
        """RK4 单步积分。"""
        k1 = self.derivatives(state, control)
        k2 = self.derivatives(state + 0.5 * dt * k1, control)
        k3 = self.derivatives(state + 0.5 * dt * k2, control)
        k4 = self.derivatives(state + dt * k3, control)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class DynamicVehicle:
    """动力学车辆适配器——与 BicycleModel 接口一致。

    内部维护 6D 状态 [x_f, y_f, yaw, vx, vy, r]，
    对外暴露 x, y, yaw, v（合速度）。
    """
    def __init__(self, params, x=0.0, y=0.0, yaw=0.0, v=0.0,
                 dt=0.02, differentiable=False):
        self.params = params
        self.dt = dt
        self.differentiable = differentiable
        self.dynamics = VehicleDynamics(params)
        self._steer_ratio = self.dynamics.steer_ratio

        # 初始 6D 状态：vx=v, vy=0, r=0
        self._state = torch.tensor(
            [float(x), float(y), float(yaw), float(v), 0.0, 0.0])

    def step(self, delta, acc):
        """前进一步。

        Args:
            delta: 前轮转角 (rad)。适配器内部乘回 steer_ratio 得方向盘角。
            acc: 加速度 (m/s²)。内部转为后驱扭矩。
        """
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(float(delta))
        if not isinstance(acc, torch.Tensor):
            acc = torch.tensor(float(acc))

        # delta(前轮转角) → 方向盘角
        delta_sw = delta * self._steer_ratio

        # acc → 后驱扭矩：T = (m * acc / 2) * R
        m = self.dynamics.m
        R = self.dynamics.R
        torque_rear = (m * acc / 2.0) * R

        control = torch.stack([
            delta_sw,
            torch.zeros_like(acc),   # T_fl = 0
            torch.zeros_like(acc),   # T_fr = 0
            torque_rear,             # T_rl
            torque_rear,             # T_rr
        ]).unsqueeze(0)  # [1, 5]

        state = self._state.unsqueeze(0)  # [1, 6]
        new_state = self.dynamics.rk4_step(state, control, self.dt)
        self._state = new_state.squeeze(0)

    def detach_state(self):
        """截断梯度链（Truncated BPTT）。"""
        self._state = self._state.detach().requires_grad_(False)

    @property
    def x(self):
        return self._state[0]

    @property
    def y(self):
        return self._state[1]

    @property
    def yaw(self):
        return self._state[2]

    @property
    def v(self):
        """合速度 = sqrt(vx² + vy²)。"""
        vx = self._state[3]
        vy = self._state[4]
        return torch.sqrt(vx * vx + vy * vy + 1e-10)

    @property
    def speed_kph(self):
        return self.v * 3.6

    @property
    def yaw_deg(self):
        return self.yaw * (180.0 / 3.141592653589793)
```

**Step 4: 运行测试确认通过**

Run: `cd sim && python -m pytest tests/test_dynamic_vehicle.py -v`
Expected: 全部 PASS

**Step 5: Commit**

```bash
git add sim/model/dynamic_vehicle.py sim/tests/test_dynamic_vehicle.py
git commit -m "[sim] 新增 DynamicVehicle 适配器：6-DOF 动力学模型，接口与 BicycleModel 一致"
```

---

### Task 2: 新建 vehicle_factory + YAML 配置

**Files:**
- Create: `sim/model/vehicle_factory.py`
- Modify: `sim/configs/default.yaml`（追加 `vehicle.model_type` 和 `dynamic_vehicle` 段）
- Test: `sim/tests/test_vehicle_factory.py`

**Step 1: 写测试文件**

```python
# sim/tests/test_vehicle_factory.py
"""车辆工厂测试。"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.vehicle import BicycleModel
from model.dynamic_vehicle import DynamicVehicle
from model.vehicle_factory import create_vehicle
from config import load_config


class TestVehicleFactory:
    def test_default_creates_kinematic(self):
        """默认配置应创建 BicycleModel。"""
        cfg = load_config()
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, BicycleModel)

    def test_kinematic_explicit(self):
        """显式指定 kinematic。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'kinematic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, BicycleModel)

    def test_dynamic_explicit(self):
        """显式指定 dynamic。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'dynamic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, DynamicVehicle)

    def test_dynamic_has_correct_interface(self):
        """dynamic 模型暴露与 kinematic 一致的接口。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'dynamic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        car.step(delta=0.0, acc=0.0)
        assert hasattr(car, 'x')
        assert hasattr(car, 'speed_kph')
        assert hasattr(car, 'yaw_deg')
        assert hasattr(car, 'detach_state')

    def test_invalid_model_type_raises(self):
        """无效 model_type 应报错。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'invalid'
        with pytest.raises(ValueError, match='model_type'):
            create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)

    def test_differentiable_forwarded(self):
        """differentiable 参数应传递到模型。"""
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'dynamic'
        car = create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        assert car.differentiable is True
```

**Step 2: 运行测试确认失败**

Run: `cd sim && python -m pytest tests/test_vehicle_factory.py -v`
Expected: FAIL

**Step 3: 修改 default.yaml**

在 `vehicle:` 段添加 `model_type: kinematic`。新增 `dynamic_vehicle:` 段：

```yaml
vehicle:
  model_type: kinematic
  wheelbase: 3.5
  steer_ratio: 17.5
  mass: 2440.0
  wheel_rolling_radius: 0.505
  windward_area: 8.0
  transmission_efficiency: 0.85
  transmission_ratio_D: 14.02

dynamic_vehicle:
  mass: 2440.0
  Iz: 9564.8
  lf: 1.354
  lr: 1.446
  wheel_radius: 0.329
  drag_coeff: 0.558
  frontal_area: 5.903
  air_density: 1.225
  rolling_coeff: 0.0065
  corner_stiff_f: 80000.0
  corner_stiff_r: 80000.0
  tire_friction_mu: 0.85
  track_width: 1.725
  steer_ratio: 16.39
```

**Step 4: 实现 vehicle_factory.py**

```python
# sim/model/vehicle_factory.py
"""车辆模型工厂：根据配置创建 BicycleModel 或 DynamicVehicle。"""
from model.vehicle import BicycleModel
from model.dynamic_vehicle import DynamicVehicle


def create_vehicle(cfg, x=0.0, y=0.0, yaw=0.0, v=0.0,
                   dt=0.02, differentiable=False):
    """根据 cfg['vehicle']['model_type'] 创建车辆模型。

    Args:
        cfg: 完整配置字典
        x, y, yaw, v: 初始状态
        dt: 仿真步长
        differentiable: 是否启用可微模式

    Returns:
        BicycleModel 或 DynamicVehicle 实例（接口一致）
    """
    model_type = cfg['vehicle'].get('model_type', 'kinematic')

    if model_type == 'kinematic':
        veh = cfg['vehicle']
        return BicycleModel(
            wheelbase=veh['wheelbase'], x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable)

    elif model_type == 'dynamic':
        dyn_params = cfg['dynamic_vehicle']
        return DynamicVehicle(
            params=dyn_params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable)

    else:
        raise ValueError(
            f"未知 vehicle.model_type: '{model_type}'，"
            f"支持: 'kinematic', 'dynamic'")
```

**Step 5: 运行测试确认通过**

Run: `cd sim && python -m pytest tests/test_vehicle_factory.py -v`
Expected: 全部 PASS

**Step 6: Commit**

```bash
git add sim/model/vehicle_factory.py sim/tests/test_vehicle_factory.py sim/configs/default.yaml
git commit -m "[sim] 新增 vehicle_factory：配置驱动的车辆模型选择（kinematic/dynamic）"
```

---

### Task 3: 改造 sim_loop.py 使用工厂

**Files:**
- Modify: `sim/sim_loop.py:10,60-62`

**Step 1: 修改 import 和创建逻辑**

变更很小，只改两处：

1. `sim_loop.py:10` — 替换 import：

```python
# 之前
from model.vehicle import BicycleModel
# 之后
from model.vehicle_factory import create_vehicle
```

2. `sim_loop.py:60-62` — 替换创建：

```python
# 之前
car = BicycleModel(wheelbase=wheelbase, x=x0, y=y0,
                   yaw=yaw0, v=init_speed, dt=dt,
                   differentiable=differentiable)
# 之后
car = create_vehicle(cfg, x=x0, y=y0, yaw=yaw0, v=init_speed,
                     dt=dt, differentiable=differentiable)
```

**Step 2: 运行全部测试确认无回归**

Run: `cd sim && python -m pytest tests/ -v`
Expected: 所有现有测试 + 新增测试全部 PASS

**Step 3: 快速验证 kinematic 模式（默认）行为不变**

Run: `cd sim && python run_demo.py --save --no-show`
Expected: 5 个场景正常运行，结果图保存到 results/

**Step 4: Commit**

```bash
git add sim/sim_loop.py
git commit -m "[sim] sim_loop 改用 vehicle_factory，支持配置切换被控对象"
```

---

### Task 4: 添加命令行 --plant 参数

**Files:**
- Modify: `sim/run_demo.py:159-170`（添加 --plant 参数，覆盖 cfg）
- Modify: `sim/optim/train.py:342-370`（添加 --plant 参数，覆盖 cfg）

**Step 1: 修改 run_demo.py**

在 `parser.add_argument` 块（约 line 161-166）中添加：

```python
parser.add_argument('--plant', type=str, default=None,
                    choices=['kinematic', 'dynamic'],
                    help='被控对象类型（覆盖 YAML 配置）')
```

在加载配置后（约 line 170）添加覆盖逻辑：

```python
cfg = load_config(args.config) if args.config else load_config()
if args.plant:
    cfg['vehicle']['model_type'] = args.plant
```

注意：原来 `cfg` 可能是 `None`（传给 `run_simulation` 时内部加载），改为始终加载。

**Step 2: 修改 train.py**

在 parser 块（约 line 360）添加：

```python
parser.add_argument('--plant', type=str, default=None,
                    choices=['kinematic', 'dynamic'],
                    help='被控对象类型（覆盖 YAML 配置）')
```

在 `train()` 函数签名中添加 `plant=None` 参数。

在 `DiffControllerParams.__init__` 中传入并应用覆盖：

```python
# train() 函数内，构造 params 前
cfg = load_config()
if plant:
    cfg['vehicle']['model_type'] = plant
params = DiffControllerParams(cfg=cfg)
```

**Step 3: 用 dynamic 模型运行 demo 验证**

Run: `cd sim && python run_demo.py --plant dynamic --save --no-show`
Expected: 5 个场景能运行完成（可能跟踪效果不如 kinematic，但不崩溃）

**Step 4: Commit**

```bash
git add sim/run_demo.py sim/optim/train.py
git commit -m "[sim] run_demo/train 添加 --plant 参数，命令行覆盖被控对象类型"
```

---

### Task 5: 端到端验证 + 文档更新

**Files:**
- Modify: `sim/CLAUDE.md`（更新模块结构和常用命令）

**Step 1: 运行全量测试**

Run: `cd sim && python -m pytest tests/ -v`
Expected: 全部 PASS

**Step 2: 双模型对比验证**

```bash
cd sim
python run_demo.py --plant kinematic --save --no-show   # 运动学基线
python run_demo.py --plant dynamic --save --no-show     # 动力学对比
```

**Step 3: 更新 sim/CLAUDE.md**

在模块结构中添加：
- `sim/model/dynamic_vehicle.py` — DynamicVehicle 适配器（6-DOF 动力学，后驱扭矩分配）
- `sim/model/vehicle_factory.py` — 工厂函数 create_vehicle（kinematic/dynamic）

在常用命令中添加：
```bash
python run_demo.py --plant dynamic --save --no-show     # 使用动力学模型
python optim/train.py --plant dynamic --epochs 50       # 用动力学模型训练
```

**Step 4: Commit + Push**

```bash
git add sim/CLAUDE.md
git commit -m "[sim] 文档更新：双被控对象集成说明"
git push
```
