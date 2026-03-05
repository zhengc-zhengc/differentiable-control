# 从可微控制项目学 PyTorch：Tensor、计算图与自动微分

> 本文档以本项目（可微控制调参）为案例，带你理解 PyTorch 的核心机制。
> 每个概念都有对应的项目代码片段，标注了文件位置和行号。

---

## 目录

1. [为什么需要可微？——问题动机](#1-为什么需要可微问题动机)
2. [PyTorch 三大核心概念](#2-pytorch-三大核心概念)
3. [项目宏观架构](#3-项目宏观架构)
4. [逐模块深入：Tensor 如何流过控制器](#4-逐模块深入tensor-如何流过控制器)
5. [不可微操作的处理技巧](#5-不可微操作的处理技巧)
6. [完整训练流程：从 loss 到参数更新](#6-完整训练流程从-loss-到参数更新)
7. [BPTT 与梯度爆炸](#7-bptt-与梯度爆炸)
8. [核心认知总结](#8-核心认知总结)

---

## 1. 为什么需要可微？——问题动机

我们有一个自动驾驶控制器，它有很多参数（查找表、PID 增益等）。传统调参方式是工程师手动试，或用黑盒优化（如网格搜索、遗传算法）。

**可微调参的思路**：如果我们能把"控制器 → 车辆模型 → 跟踪误差"这整条链路变成一个可求导的函数，就可以用梯度下降来自动优化参数——就像训练神经网络一样。

```
传统方式：参数 → [黑盒控制器] → 误差    （只能看到误差大小，不知道往哪个方向调）
可微方式：参数 → [可微控制器] → 误差    （知道每个参数对误差的影响方向和大小）
                                         ↑ 这就是"梯度"
```

**关键问题**：控制器代码里充满了 `if/else`、`clamp`、`sign` 这些不可微操作。怎么办？

**回答**：只要把计算过程转化为 PyTorch 的 tensor 运算，PyTorch 就能自动帮你求导。对于不可微的操作，我们用**平滑近似**或**梯度穿透（STE）**来处理。

---

## 2. PyTorch 三大核心概念

### 2.1 Tensor：带计算历史的多维数组

`torch.Tensor` 不只是一个数值容器（类似 numpy 数组），它还能记住自己是**怎么被算出来的**。

```python
import torch

# 创建一个普通 tensor
a = torch.tensor(3.0)                # 标量 tensor，值为 3.0
b = torch.tensor(4.0)                # 标量 tensor，值为 4.0

# 运算产生新 tensor
c = a * b + torch.sin(a)             # c = 12 + sin(3) ≈ 12.14
# c 内部记住了：c = a*b + sin(a)
```

**项目中的体现**：车辆的状态 `(x, y, yaw, v)` 全部是 `torch.Tensor`。

```python
# sim/model/vehicle.py:15-19
self.x = torch.tensor(float(x))      # 车辆位置 x
self.y = torch.tensor(float(y))      # 车辆位置 y
self.yaw = torch.tensor(float(yaw))  # 车辆航向角
self.v = torch.tensor(float(v))      # 车辆速度
```

### 2.2 计算图：tensor 之间的运算关系

每次对 tensor 做运算，PyTorch 都在后台构建一个**计算图**（computational graph），记录运算的依赖关系。

```
          a ──→ [*] ──→ [+] ──→ c
          b ──↗        ↗
          a ──→ [sin] ─┘
```

**项目中的体现**：车辆模型的 `step()` 就是在不断往计算图里添加节点。

```python
# sim/model/vehicle.py:28-31
# 每执行一步，就会产生新的 tensor 节点，挂在计算图上
self.x = self.x + self.v * torch.cos(self.yaw) * self.dt
self.y = self.y + self.v * torch.sin(self.yaw) * self.dt
self.yaw = self.yaw + self.v * torch.tan(delta) / self.L * self.dt
self.v = self.v + acc * self.dt
```

每一步执行后，新的 `self.x` 依赖于旧的 `self.x`、`self.v`、`self.yaw`。
如果仿真跑 1000 步，计算图就是 1000 步串联的长链——这正是 BPTT（Backpropagation Through Time）的基础。

### 2.3 自动微分（Autograd）：反向传播计算梯度

给定一个标量 loss（如跟踪误差），PyTorch 可以沿着计算图**反向**走一遍，自动算出 loss 对每个参数的偏导数。

```python
# 手动验证自动微分
a = torch.tensor(3.0, requires_grad=True)   # 告诉 PyTorch：我要对 a 求导
b = torch.tensor(4.0, requires_grad=True)

c = a * b + torch.sin(a)    # c = a*b + sin(a)
c.backward()                 # 反向传播！

print(a.grad)  # dc/da = b + cos(a) = 4 + cos(3) ≈ 3.01
print(b.grad)  # dc/db = a = 3.0
```

`requires_grad=True` 是开关。在本项目中，`nn.Parameter` 自动开启这个开关。

---

## 3. 项目宏观架构

### 3.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        训练循环（train.py）                              │
│                                                                         │
│  ┌───────────┐                                                          │
│  │ 参考轨迹   │  trajectory.py 生成圆弧/正弦/组合轨迹                     │
│  └─────┬─────┘                                                          │
│        ↓                                                                │
│  ┌─────────────────────────────  50Hz 仿真循环（sim_loop.py）──────────┐ │
│  │                                                                     │ │
│  │  ┌──────────┐  车辆状态   ┌─────────────────┐  steer   ┌─────────┐ │ │
│  │  │ 车辆模型  │──────────→│ 横向控制器        │────────→│         │ │ │
│  │  │ vehicle  │  (x,y,v,  │ lat_truck        │         │ 车辆模型 │ │ │
│  │  │ .py      │   yaw)    │ (nn.Module)      │         │ .step() │ │ │
│  │  │          │           └─────────────────┘         │         │ │ │
│  │  │          │  车辆状态   ┌─────────────────┐  acc    │         │ │ │
│  │  │          │──────────→│ 纵向控制器        │────────→│         │ │ │
│  │  │          │           │ lon.py           │         │         │ │ │
│  │  │          │           │ (nn.Module)      │         │         │ │ │
│  │  │          │           └─────────────────┘         │         │ │ │
│  │  │          │←──────────────────────────────────────┘         │ │ │
│  │  └──────────┘                                                   │ │
│  │       ↓ 每步记录 lateral_error, heading_error, speed 等          │ │
│  │  ┌──────────┐                                                   │ │
│  │  │ history  │  tensor 列表（保持在计算图中）                       │ │
│  │  └────┬─────┘                                                   │ │
│  └───────┼─────────────────────────────────────────────────────────┘ │
│          ↓                                                           │
│  ┌──────────────┐                                                    │
│  │ tracking_loss │  MSE(横向误差) + MSE(航向误差) + MSE(速度误差) + 平滑度│
│  └──────┬───────┘                                                    │
│         ↓                                                            │
│  loss.backward()     ← 反向传播：沿计算图回溯，算出所有 nn.Parameter 的梯度│
│         ↓                                                            │
│  optimizer.step()    ← Adam 优化器用梯度更新参数                        │
│         ↓                                                            │
│  重复 N epochs                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 文件结构与职责

| 文件 | 职责 | 核心 torch 概念 |
|------|------|----------------|
| `sim/common.py` | 基础运算库 | `torch.autograd.Function`（自定义反向传播） |
| `sim/config.py` | 配置加载/保存 | `torch.Tensor` 与 Python 类型互转 |
| `sim/model/vehicle.py` | 自行车模型 | 计算图的**链式构建** |
| `sim/model/trajectory.py` | 轨迹生成与查询 | `torch.no_grad()`、`detach()` |
| `sim/controller/lat_truck.py` | 横向控制器 | `nn.Module`、`nn.Parameter`、`register_buffer` |
| `sim/controller/lon.py` | 纵向控制器 | `nn.Parameter`、`smooth_step` 替代 if/else |
| `sim/sim_loop.py` | 闭环仿真循环 | BPTT、`detach_state()`（截断梯度链） |
| `sim/optim/train.py` | 训练入口 | `backward()`、`optimizer`、`clip_grad_norm_` |

---

## 4. 逐模块深入：Tensor 如何流过控制器

### 4.1 nn.Module 与 nn.Parameter——"可训练的模块"

**概念**：`nn.Module` 是 PyTorch 中"一切可训练模块"的基类。它干两件事：
1. **管理参数**：通过 `nn.Parameter` 声明哪些 tensor 是可训练参数
2. **组织计算**：定义 forward 计算逻辑

**`nn.Parameter` vs `register_buffer`**：

| 类型 | `requires_grad` | 随 `model.parameters()` 导出 | 用途 |
|------|-----------------|------------------------------|------|
| `nn.Parameter` | **True**（自动） | **是** | 需要梯度优化的参数 |
| `register_buffer` | **False** | 否（但随模型保存/加载） | 固定常量、不需要优化的参数 |

**项目代码**：横向控制器的参数声明。

```python
# sim/controller/lat_truck.py:18-62

class LatControllerTruck(nn.Module):      # 继承 nn.Module，获得参数管理能力

    def __init__(self, cfg: dict, differentiable: bool = False):
        super().__init__()                # 必须调用父类 __init__

        # ── 固定参数：register_buffer ──
        # kLh 是车辆物理属性，不应该被优化器修改
        self.register_buffer('kLh', torch.tensor(float(lat['kLh'])))

        # T1 是安全约束（航向误差上限），也不应该被优化
        # xs 是查找表的 x 轴（速度断点），永远固定
        # ys 是查找表的 y 值
        xs, ys = table_from_config(lat['T1_max_theta_deg'])
        self.register_buffer('T1_x', xs)   # x 轴：固定断点
        self.register_buffer('T1_y', ys)   # y 值：固定（安全参数）

        # ── 可训练参数：nn.Parameter ──
        # T2 是预瞄距离时间系数，属于"控制设计参数"，可以优化
        xs, ys = table_from_config(lat['T2_prev_time_dist'])
        self.register_buffer('T2_x', xs)   # x 轴：固定断点
        # 关键！T2_y 用 nn.Parameter 包装，优化器会更新它
        setattr(self, 'T2_y', nn.Parameter(ys))
```

**理解要点**：
- `nn.Parameter(tensor)` 做两件事：① `requires_grad=True`（开启梯度追踪）② 注册到模块的参数列表中
- `register_buffer` 的 tensor **不会**被 `model.parameters()` 返回，所以优化器不会碰它
- 查找表的 **x 轴**（速度断点，如 `[0, 20, 40, 60]`）始终是 buffer，因为我们不想改变"在哪些速度下查表"
- 查找表的 **y 值**（具体数值，如 `[1.0, 0.8, 0.6, 0.5]`）如果是控制设计参数，就用 `nn.Parameter` 使其可训练

### 4.2 lookup1d——可微的分段线性插值

这是整个项目中**最基础的可微操作**：给定一张查找表和查询值，用线性插值算出结果。

```python
# sim/common.py:7-25

def lookup1d(table_x: torch.Tensor, table_y: torch.Tensor,
             x: torch.Tensor) -> torch.Tensor:
    """分段线性插值，边界 clamp。"""

    # ① 将查询值限制在表的 x 范围内
    x_clamped = torch.clamp(x, table_x[0].item(), table_x[-1].item())

    # ② 找到 x 落在哪个区间 [table_x[i], table_x[i+1]]
    idx = torch.searchsorted(table_x, x_clamped) - 1
    idx = torch.clamp(idx, 0, len(table_x) - 2)
    i = idx.long()

    # ③ 取出区间端点
    x0, x1 = table_x[i], table_x[i + 1]
    y0, y1 = table_y[i], table_y[i + 1]    # ← 如果 table_y 是 nn.Parameter，
                                             #   梯度会流到这里！

    # ④ 线性插值：y = y0 + (y1 - y0) * t
    t = (x_clamped - x0) / (x1 - x0 + 1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    return y0 + (y1 - y0) * t
    #      ↑ 这个运算对 y0、y1 可微
    #        dy/d(y0) = 1 - t
    #        dy/d(y1) = t
```

**关键认知**：线性插值 `y = y0 + (y1 - y0) * t` 天然可微。如果 `y0`、`y1` 来自 `nn.Parameter`（查找表的 y 值），梯度会通过这个插值公式自动流回去。这就是为什么**查找表都能被优化**——只要 y 值是 `nn.Parameter`，插值运算就是可微的。

### 4.3 BicycleModel——计算图的链式增长

自行车模型是整个系统的"物理模拟器"。每调用一次 `step()`，计算图就往后延伸一步。

```python
# sim/model/vehicle.py:22-35

def step(self, delta, acc):
    """前进一步。delta: 前轮转角(rad), acc: 加速度(m/s^2)。"""

    # 运动学方程——每一行都是一次 tensor 运算，自动记入计算图
    self.x = self.x + self.v * torch.cos(self.yaw) * self.dt
    self.y = self.y + self.v * torch.sin(self.yaw) * self.dt
    self.yaw = self.yaw + self.v * torch.tan(delta) / self.L * self.dt
    self.v = self.v + acc * self.dt

    # 速度下限保护
    if self.differentiable:
        # softplus 是 ReLU 的平滑版本：softplus(x) ≈ max(x, 0)
        # 但在 x=0 附近是光滑的，导数连续，不会产生梯度断裂
        self.v = torch.nn.functional.softplus(self.v, beta=10.0)
    else:
        # clamp 在 v=0 处导数为 0（梯度消失），非可微路径无所谓
        self.v = torch.clamp(self.v, min=0.0)
```

**计算图可视化（3 步仿真）**：

```
step 0:                    step 1:                    step 2:
  x₀ ─→ x₁                 x₁ ─→ x₂                 x₂ ─→ x₃
  v₀ ─↗  ↑                 v₁ ─↗  ↑                 v₂ ─↗
  yaw₀─↗  ↑                yaw₁─↗  ↑                yaw₂─↗
  delta₀─↗                 delta₁─↗                 delta₂─↗
  acc₀ ─→ v₁              acc₁ ─→ v₂              acc₂ ─→ v₃

  整个链条连成一条长线：loss 可以一路回溯到 step 0 的 delta₀ 和 acc₀，
  进而回溯到产生 delta₀ 的控制器参数。
```

**`detach_state()` 的作用——截断梯度链**：

```python
# sim/model/vehicle.py:37-42

def detach_state(self):
    """截断梯度链：将当前状态从计算图中 detach。"""
    self.x = self.x.detach().requires_grad_(False)
    self.y = self.y.detach().requires_grad_(False)
    self.yaw = self.yaw.detach().requires_grad_(False)
    self.v = self.v.detach().requires_grad_(False)
```

`detach()` 的意思是：**切断这个 tensor 与之前计算图的连接**。新产生的 tensor 数值不变，但 PyTorch 不再记得它是怎么算出来的。这用于 Truncated BPTT——如果仿真跑 1000 步但只想让梯度回传 64 步，就每 64 步 detach 一次。

### 4.4 横向控制器——完整的可微计算流程

横向控制器是项目中最复杂的可微组件。它有两条计算路径：

| 路径 | 方法 | 中间值类型 | 用途 |
|------|------|-----------|------|
| V1（不可微） | `_compute_v1` | Python `float` | 验证、可视化 |
| V2（可微） | `_compute_differentiable` | `torch.Tensor` | 训练 |

**V1 路径**中大量使用 `.item()` 将 tensor 转为 float——这会**切断计算图**（因为 float 不在 PyTorch 的追踪范围内）：

```python
# sim/controller/lat_truck.py:134 (V1 路径)
max_theta_deg = lookup1d(self.T1_x, self.T1_y, speed_t).item()
#                                                          ↑
#                                        .item() 把 tensor 变成 float
#                                        计算图到此为止，梯度无法回传
```

**V2 路径**则全程保持 tensor 运算：

```python
# sim/controller/lat_truck.py:232 (V2 路径)
max_theta_deg = lookup1d(self.T1_x, self.T1_y, speed_kph_safe)
#                                                              ↑
#                                        返回 tensor，计算图保持连续
#                                        如果 T1_y 是 nn.Parameter，梯度可以流过
```

下面跟着 V2 路径走一遍控制器的 10 步算法，重点看 tensor 是怎么流动的：

```python
# sim/controller/lat_truck.py:202-310 (可微路径，节选关键步骤)

def _compute_differentiable(self, x, y, yaw_deg, speed_kph, ...):

    # ── Step 1: 查表 ──
    # lookup1d 返回的全是 tensor。T2_y 是 nn.Parameter，
    # 所以 prev_time_dist 的计算图连着 T2_y
    prev_time_dist = lookup1d(self.T2_x, self.T2_y, speed_kph_safe)
    #                                   ↑ nn.Parameter
    #                         梯度路径：loss → ... → prev_time_dist → T2_y

    # T5/T6 也是 nn.Parameter，梯度通过查表结果传回去
    near_pt_time = lookup1d(self.T5_x, self.T5_y, speed_kph_safe)
    far_pt_time = lookup1d(self.T6_x, self.T6_y, speed_kph_safe)

    # ── Step 2: 轨迹查询 ──
    # query_nearest_by_position 用 detached argmin（不可微），
    # 但返回的轨迹点信息足够后续计算使用
    currt = analyzer.query_nearest_by_position(x, y)

    # 关键！可微时间查询：T5/T6 的梯度通过此路径回传
    near_kappa, _, _, _ = analyzer.query_by_time_differentiable(
        t_base + near_pt_time)
    #                 ↑ near_pt_time 来自 T5_y (nn.Parameter)
    #                   + 号保持了梯度连接
    #                   query_by_time_differentiable 内部用 lookup1d 插值
    #                   所以 near_kappa 对 near_pt_time 可微
    #                   进而对 T5_y 可微！

    # ── Step 3: 误差计算（纯 tensor 运算，天然可微）──
    lateral_error = cos_theta * dy - sin_theta * dx
    heading_error = normalize_angle(yaw_rad - currt.theta)
    #               ↑ normalize_angle 用 atan2(sin, cos) 实现，可微

    # ── Step 6: target_theta ──
    # 这里用 STE clamp 替代 sign * min(|x|, L)
    # 详见第 5 节"不可微操作的处理"
    target_theta = _straight_through_clamp(
        error_angle_raw, -max_err_angle_limit, max_err_angle_limit)

    # ── Step 8-9: 反馈/前馈转向角 ──
    steer_fb = rate_limit(self.steer_fb_prev, steer_fb_raw,
                          self.rate_limit_fb, dt, differentiable=True)
    #                                              ↑ 使用 STE 版 rate_limit
    self.steer_fb_prev = steer_fb.detach().clone()
    #                            ↑ detach！状态保存时切断梯度链
    #                              否则下一步会通过 prev 追溯到更早的计算图

    # ── Step 10: 合并输出 ──
    steer_raw = smooth_clamp(steer_fb + steer_ff,
                             -max_steer_angle, max_steer_angle, temp=1.0)
    #                                                           ↑ temp=1.0
    #           之前用 temp=0.1 导致梯度爆炸，提升后导数从 10x 降到 ~1x

    return (steer_out, ...)
```

### 4.5 纵向控制器——smooth_step 替代 if/else

C++ 原始代码用大量 `if/else` 根据速度选择不同参数：

```cpp
// C++ 原始逻辑（不可微）
if (speed < switch_speed) {
    kp = low_speed_kp;
} else {
    kp = high_speed_kp;
}
```

**问题**：`if/else` 是离散分支，PyTorch 的计算图只走一条路径，另一条路径的参数完全收不到梯度。

**解决**：用 `smooth_step`（sigmoid 平滑阶跃函数）做加权混合：

```python
# sim/controller/lon.py:307-310

# sigmoid 是 S 形曲线：接近 0 表示"远低于阈值"，接近 1 表示"远高于阈值"
# 在阈值附近是平滑过渡，两边参数都能收到梯度
w_low = 1.0 - smooth_step(speed_mps, self.switch_speed, temp=0.5)
# w_low ≈ 1 when speed << switch_speed (低速)
# w_low ≈ 0 when speed >> switch_speed (高速)

# 加权混合：两个分支的参数都参与运算，都能收到梯度
kp = w_low * self.low_speed_kp + (1.0 - w_low) * self.high_speed_kp
ki = w_low * self.low_speed_ki + (1.0 - w_low) * self.high_speed_ki
```

`smooth_step` 的实现很简单：

```python
# sim/common.py:81-86

def smooth_step(x: torch.Tensor, threshold, temp: float = 1.0) -> torch.Tensor:
    """sigmoid 平滑阶跃函数。"""
    return torch.sigmoid((x - threshold) / temp)
    #      ↑ sigmoid(z) = 1 / (1 + exp(-z))
    #        temp 控制过渡的锐利程度：
    #        temp 大 → 过渡缓慢（梯度小，数值稳定）
    #        temp 小 → 过渡陡峭（梯度大，接近硬 if/else）
```

### 4.6 PID 控制器——经典控制也能可微

PID 控制器本质上就是加减乘除 + 积分（累加）+ 微分（差分），天然可微：

```python
# sim/common.py:189-200

def control(self, error, dt, kp, ki, kd,
            integrator_enable, sat, differentiable=False):
    """计算 PID 输出。kp/ki/kd/sat 从外部传入（可为 nn.Parameter）。"""

    # 积分项：累加误差
    if integrator_enable:
        self.integral = clamp(self.integral + error * dt, -sat, sat,
                              differentiable)
        # ↑ error 是 tensor, dt 是 float
        #   error * dt 是 tensor 运算，保持在计算图中
        #   如果 differentiable=True，clamp 用 smooth_clamp（可微）

    # 微分项：差分
    derivative = (error - self.prev_error) / dt

    # prev_error 要 detach！否则梯度会穿越时间步
    self.prev_error = error.detach().clone()

    # 输出：纯线性组合，天然可微
    return kp * error + ki * self.integral + kd * derivative
    #      ↑ 如果 kp 是 nn.Parameter，这里 kp * error 就是
    #        "nn.Parameter 乘以 tensor"，梯度自动流向 kp
```

**理解要点**：PID 的 `kp * error` 中，`kp` 来自 `nn.Parameter`，`error` 来自仿真中的 tensor 运算链。这个乘法让 loss 的梯度可以一路传到 `kp`，优化器就能调节 `kp` 的值。

### 4.7 TrajectoryAnalyzer——哪些查询可微，哪些不可微

轨迹分析器需要回答"离车辆最近的轨迹点在哪"。这里有个重要的可微性问题：

**不可微的操作——argmin（最近点搜索）**：

```python
# sim/model/trajectory.py:152-167

def query_nearest_by_position(self, x, y) -> TrajectoryPoint:
    """最近点查询（detached argmin）。"""

    # 用 detach 切断梯度！
    if isinstance(x, torch.Tensor):
        x_val = x.detach()    # ← 不让梯度通过 argmin
    ...

    # torch.no_grad() 包裹整个计算：不构建计算图
    with torch.no_grad():
        dx = self._xs - x_val
        dy = self._ys - y_val
        dists = dx * dx + dy * dy
        idx = torch.argmin(dists).item()  # ← argmin 不可微！
    #       ↑ argmin 是"选择最小值的索引"
    #         它的导数没有意义（索引是整数，对输入不连续）
    #         所以用 no_grad + detach 完全隔离

    return self.points[idx]
```

**为什么 argmin 不可微？** 想象 `argmin([3.0, 1.0, 2.0])` = 1（索引1的值最小）。如果你把 `3.0` 微调成 `3.001`，结果还是 1；但如果 `1.0` 微调成 `2.5`，结果突变为 2。这种"突变"意味着不可微。

**可微的操作——时间插值查询**：

```python
# sim/model/trajectory.py:198-207

def query_by_time_differentiable(self, t: torch.Tensor):
    """可微时间查询——用 lookup1d 插值，对 t 可微。"""

    kappa = lookup1d(self._t_arr, self._kappa_arr, t)
    # ↑ 给定时间 t，在"时间→曲率"表中做线性插值
    #   因为线性插值对 t 可微，
    #   所以如果 t = t_base + near_pt_time（其中 near_pt_time 来自 T5_y），
    #   梯度可以从 kappa 一路流回 T5_y
    ...
```

**总结**：
- "从哪个轨迹点开始"（位置 → 索引）→ 不可微，用 `detach`
- "从该点出发，经过 Δt 后轨迹的属性是什么"（时间 → 插值）→ 可微，用 `lookup1d`

---

## 5. 不可微操作的处理技巧

### 5.1 问题清单

控制器代码里有四类不可微操作：

| 操作 | 不可微原因 | 出现位置 |
|------|-----------|---------|
| `clamp(x, lo, hi)` | 边界处导数为 0 | rate limiter, 误差限幅 |
| `sign(x)` | x=0 处不连续 | Step 6 方向判断 |
| `if/else` | 只走一条路径 | 低速/高速 PID 切换 |
| `argmin` | 结果是整数索引 | 最近点查询 |

### 5.2 技巧一：smooth 平滑近似

**思路**：用光滑函数近似不可微函数，使其"几乎一样"但处处有导数。

```python
# sim/common.py:30-44

def smooth_clamp(x, lo, hi, temp=0.1):
    """用 tanh 平滑 clamp。temp 控制平滑程度。"""
    mid = (lo + hi) / 2.0       # 区间中点
    half = (hi - lo) / 2.0      # 区间半宽
    return mid + half * torch.tanh((x - mid) / (half * temp))
    #                   ↑ tanh 在 (-1,1) 之间，导数处处非零
    #                     当 x 远离边界时 tanh ≈ ±1（饱和），近似硬 clamp
    #                     当 x 在边界附近时，tanh 的导数 ≈ 1/temp
    #                     temp 越小，越接近硬 clamp，但导数越大
```

**直觉**：想象硬 clamp 是一个方角的"U"型管道，smooth_clamp 把方角磨圆了。圆角处水流可以顺畅通过（梯度可以传播），而方角处水流会被卡住（梯度为零）。

### 5.3 技巧二：Straight-Through Estimator（STE，梯度穿透）

**思路**：前向计算用**硬操作**（精确），反向传梯度时**假装没有这个操作**（让梯度直接穿过）。

这是本项目中最精巧的技巧，用自定义 `torch.autograd.Function` 实现：

```python
# sim/common.py:136-171

class _StraightThroughClamp(torch.autograd.Function):
    """Straight-through estimator for clamp.
    前向：硬 clamp（精确限幅）。
    反向：梯度无条件传给 x（假装没有 clamp）。"""

    @staticmethod
    def forward(ctx, x, lo, hi):
        # ctx 是"上下文"，用于在 forward 和 backward 之间传递信息
        lo_t = torch.as_tensor(lo)
        hi_t = torch.as_tensor(hi)
        ctx.save_for_backward(x, lo_t, hi_t)
        return torch.clamp(x, lo_t.item(), hi_t.item())
        #      ↑ 前向正常做硬 clamp，结果精确

    @staticmethod
    def backward(ctx, grad_output):
        x, lo, hi = ctx.saved_tensors
        # 关键！x 总是收到梯度，不管前向是否被 clamp 了
        grad_x = grad_output           # ← "straight-through"
        # lo/hi 在触发限幅时也收到梯度（如果 lo/hi 是 nn.Parameter）
        grad_lo = grad_output if x.item() <= lo.item() else None
        grad_hi = grad_output if x.item() >= hi.item() else None
        return grad_x, grad_lo, grad_hi
```

**为什么需要 STE？** 考虑 rate limiter（速率限制器）：

```
目标值 target = 100
前一步 prev = 0
最大变化量 max_delta = 5

标准 clamp: output = prev + clamp(target - prev, -5, 5) = 5
导数: d(output)/d(target) = 0    ← 因为 delta 被 clamp 到上界了
                                    梯度消失！优化器完全不知道 target 想去 100

STE clamp: 前向 output 同样 = 5（精确结果不变）
           反向 d(output)/d(target) = 1   ← 假装没有 clamp
           优化器知道"如果 target 增大，output 想增大"
```

**项目中的使用**：

```python
# sim/common.py:117-133

def rate_limit(prev, target, rate, dt, differentiable=False):
    max_delta = rate * dt
    delta = target - prev
    if differentiable:
        # STE clamp：前向精确限幅，反向梯度穿透
        clamped_delta = _straight_through_clamp(delta, -max_delta, max_delta)
    else:
        clamped_delta = torch.clamp(delta, -max_delta, max_delta)
    return prev + clamped_delta
```

### 5.4 技巧三：detach 隔离

对于完全不可微的操作（如 argmin），直接用 `detach()` 把它从计算图中隔离出来。这意味着**放弃该操作的梯度**，但保留它的前向计算结果。

```python
# sim/model/trajectory.py:155-156
x_val = x.detach()     # ← 告诉 PyTorch：argmin 的输入不需要追踪梯度
                        #   x 本身在计算图中的其他用途不受影响

# sim/model/vehicle.py:39-42（detach_state 用于截断 BPTT）
self.x = self.x.detach().requires_grad_(False)
#                ↑ detach：切断与之前计算的连接
#                         ↑ requires_grad_(False)：不再追踪后续运算
```

### 5.5 技巧四：数学等价化简

有时不需要复杂的 smooth 近似——先**化简数学表达式**，可能发现原始操作其实是可微的。

```python
# 原始 C++ 代码（Step 6）：
abs_error = abs(error_angle_raw)
max_err_angle = min(abs_error, max_theta_limit)
target_theta = sign(error_angle_raw) * max_err_angle
# ↑ 包含 abs、min、sign 三个不可微操作

# 数学等价化简后：
# sign(x) * min(|x|, L)  ≡  clamp(x, -L, L)
target_theta = clamp(error_angle_raw, -max_theta_limit, max_theta_limit)
# ↑ 只需要一个 STE clamp，导数恒为 1，不会导致梯度爆炸
```

这个化简在项目中的实际代码：

```python
# sim/controller/lat_truck.py:274-276
target_theta = _straight_through_clamp(
    error_angle_raw, -max_err_angle_limit, max_err_angle_limit)
```

---

## 6. 完整训练流程：从 loss 到参数更新

### 6.1 训练入口：DiffControllerParams

```python
# sim/optim/train.py:27-36

class DiffControllerParams(nn.Module):
    """封装横向 + 纵向控制器为一个 nn.Module。"""

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        # 两个子 Module，它们的 nn.Parameter 自动注册到父 Module
        self.lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        self.lon_ctrl = LonController(cfg, differentiable=True)
        # 此时 self.parameters() 返回所有可训练参数：
        # lat_ctrl 的 T2_y, T3_y, T4_y, T5_y, T6_y
        # lon_ctrl 的 station_kp, station_ki, low_speed_kp, ...
```

**nn.Module 的嵌套**：在 PyTorch 中，nn.Module 可以包含其他 nn.Module。调用父模块的 `.parameters()` 会递归收集所有子模块的参数。

### 6.2 tracking_loss——将仿真结果转为标量 loss

```python
# sim/optim/train.py:74-125

def tracking_loss(history, ref_speed, w_lat=10.0, w_head=5.0, ...):
    # ① 从 history（tensor 列表）中提取各维度误差
    lat_errs = torch.stack([h['lateral_error'] for h in history])
    #          ↑ torch.stack：把多个标量 tensor 堆叠成一个向量
    #            每个 h['lateral_error'] 都在计算图中
    #            stack 后的整个向量也在计算图中

    head_errs = torch.stack([h['heading_error'] for h in history])
    speeds = torch.stack([h['v'] for h in history])

    # ② 计算 MSE（均方误差）
    lat_mse = (lat_errs ** 2).mean()
    #          ↑ 平方、求均值——都是标准 tensor 运算，可微

    # ③ 加权求和得到总 loss
    loss = w_lat * lat_mse + w_head * head_mse + w_speed * speed_mse
    #      ↑ loss 是标量 tensor，连着整个计算图

    # ④ 平滑度惩罚：鼓励输出不要剧烈变化
    steer_rate = steers[1:] - steers[:-1]  # 相邻步转向角的差
    loss = loss + w_steer_rate * (steer_rate ** 2).mean()

    return loss
```

### 6.3 训练循环——梯度下降的完整流程

```python
# sim/optim/train.py:139-251 (简化版)

def train(...):
    params = DiffControllerParams()       # 创建可训练参数模块

    # ── 创建优化器 ──
    # Adam 是一种自适应学习率的梯度下降算法
    # 分组学习率：查找表用较低 lr（它们是多维的，更敏感）
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': table_params, 'lr': lr_tables},
    ])

    for epoch in range(n_epochs):

        # ① 清零梯度（PyTorch 默认累积梯度，每个 epoch 开始前必须清零）
        optimizer.zero_grad()

        epoch_loss = torch.tensor(0.0)

        for traj_name in trajectories:
            traj = builder(sim_speed)

            # ② 前向传播：跑一次闭环仿真，构建完整计算图
            history = run_simulation(
                traj, cfg=params.cfg,
                lat_ctrl=params.lat_ctrl,     # 传入可训练的控制器
                lon_ctrl=params.lon_ctrl,
                differentiable=True,          # 全程 tensor 运算
                tbptt_k=tbptt_k)              # 每 K 步截断梯度

            # ③ 计算 loss
            loss, details = tracking_loss(history, ref_speed=sim_speed)
            epoch_loss = epoch_loss + loss

        epoch_loss = epoch_loss / len(trajectories)

        # ④ 反向传播：PyTorch 自动沿计算图反向传播，算出所有参数的梯度
        epoch_loss.backward()
        # 此时 params.lat_ctrl.T2_y.grad 已经有值了！
        # 它告诉你"T2_y 增大一点，loss 会增大多少"

        # ⑤ 梯度裁剪：防止极端梯度导致参数更新过大
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params.parameters(), max_norm=grad_clip)

        # ⑥ 参数更新：优化器根据梯度调整参数
        optimizer.step()
        # 此时 T2_y 的值已经被 Adam 更新了
        # 新值 = 旧值 - lr * (梯度 / 自适应缩放)
```

**一句话理解训练循环**：前向跑仿真 → 算 loss → 反向传梯度 → 裁剪梯度 → 更新参数 → 重复。

### 6.4 梯度路径：从 loss 到 nn.Parameter

以横向控制器 T2_y（预瞄距离时间系数）为例，梯度是怎么从 loss 一路传回来的：

```
loss
 ↑ (lat_mse = (lat_errs^2).mean())
 ↑
lateral_error = cos(θ) * dy - sin(θ) * dx
 ↑ (dx = car.x - ref.x, dy = car.y - ref.y)
 ↑
car.x = car.x_prev + car.v * cos(car.yaw) * dt     ← vehicle.step()
 ↑
car.yaw = car.yaw_prev + car.v * tan(delta) / L * dt  ← vehicle.step()
 ↑
delta = steer_out / steer_ratio * DEG2RAD
 ↑
steer_out ← rate_limit(prev, steer_raw, ...)         ← STE，梯度穿透
 ↑
steer_raw ← smooth_clamp(steer_fb + steer_ff, ...)
 ↑
steer_fb_raw = atan(target_curvature * wheelbase) * ratio * slip
 ↑
target_curvature = (target_theta - real_theta + ...) / denom
 ↑
denom = smooth_lower_bound(reach_time_theta * speed, ...)
 ↑
reach_time_theta = lookup1d(T3_x, T3_y, speed)       ← T3_y 是 nn.Parameter！
...
prev_dist = smooth_lower_bound(speed * prev_time_dist, ...)
 ↑
prev_time_dist = lookup1d(T2_x, T2_y, speed)          ← T2_y 是 nn.Parameter！
                                 ↑
                        梯度最终到达这里：T2_y.grad 被填入
```

---

## 7. BPTT 与梯度爆炸

### 7.1 什么是 BPTT

BPTT（Backpropagation Through Time）就是在时间序列上做反向传播。在闭环仿真中，每一步的输出是下一步的输入，形成长链：

```
state₀ → controller → vehicle → state₁ → controller → vehicle → state₂ → ...
```

反向传播时，梯度需要沿着这条链一步步往回传。链条越长，中间每一步的"放大系数"连乘就越多。

### 7.2 梯度爆炸的根因

如果某步的局部导数 > 1，连乘 N 步就指数增长：

```
每步放大 2x  → 64步后：2^64 = 1.8 × 10^19（勉强可用，配合 grad_clip）
每步放大 10x → 64步后：10^64（Inf，直接溢出）
```

本项目中 `smooth_sign(x, temp=0.01)` 在 x=0 附近的导数 = 1/0.01 = **100**。如果信号恰好经过零点，这个 100 倍放大就会被链式乘法捕获。

### 7.3 Truncated BPTT——截断梯度链

```python
# sim/sim_loop.py:83-86

# 每 K 步 detach 一次：切断计算图，限制梯度回传的最大距离
if tbptt_k > 0 and step > 0 and step % tbptt_k == 0:
    car.detach_state()        # 车辆状态与之前的计算图断开
    prev_steer = prev_steer.detach()  # 上一步转向角也断开
```

**效果**：如果 `tbptt_k=64`（仿真步长 0.02s → 1.28s 窗口），梯度最多只往回传 64 步。这意味着每次参数更新只考虑"最近 1.28 秒的影响"，但避免了长链的指数增长。

### 7.4 本项目的梯度爆炸修复策略

| 策略 | 代码位置 | 效果 |
|------|---------|------|
| 数学化简 | `lat_truck.py:275` STE clamp 替代 smooth_sign | 消除最大梯度源（100x → 1x） |
| 提升 temperature | `lat_truck.py:260,303` temp 0.1→1.0 | 降低局部导数（10x → ~1x） |
| 截断 BPTT | `sim_loop.py:84` tbptt_k=64 | 限制链式乘法长度 |
| 梯度裁剪 | `train.py:236` clip_grad_norm_ | 安全网，防止残余异常梯度 |
| 梯度钩子 | `train.py:166-170` register_hook 清理 NaN | 保护 Adam 的二阶矩 |

---

## 8. 核心认知总结

### 8.1 什么样的计算可以自动微分？

**规则很简单**：只要你的计算**全程使用 `torch.Tensor` 做运算**，PyTorch 就能自动微分。

具体来说：
- 所有标准数学运算（加减乘除、三角函数、指数对数）→ 可微
- 线性插值 → 可微
- `torch.sigmoid`、`torch.tanh`、`softplus` → 可微
- `torch.clamp` → 可微（但边界处导数为 0，梯度会消失）
- `argmin`、`argmax` → **不可微**（结果是整数索引）
- `.item()` → **切断计算图**（tensor → float，PyTorch 不再追踪）
- Python `if/else` → **只走一条路径**，另一条路径的参数无梯度

### 8.2 "可微化"一段代码的步骤

1. **所有中间变量用 tensor**——不要 `.item()` 转 float
2. **if/else 用 smooth_step 混合**——两条路径的参数都参与运算
3. **clamp 用 smooth_clamp 或 STE clamp**——边界处梯度不消失
4. **sign 能化简就化简**——`sign(x)*min(|x|,L)` = `clamp(x,-L,L)`
5. **argmin 用 detach 隔离**——放弃其梯度，用其前向结果
6. **控制 smooth 近似的 temperature**——确保局部导数 ≤ 2-3x

### 8.3 关键公式速查

| 概念 | 公式 | PyTorch |
|------|------|---------|
| 线性插值 | `y = y0 + (y1-y0)*t` | `lookup1d()` |
| 平滑 clamp | `mid + half*tanh((x-mid)/(half*temp))` | `smooth_clamp()` |
| 平滑阶跃 | `sigmoid((x-threshold)/temp)` | `smooth_step()` |
| 平滑下界 | `lo + softplus((x-lo)*k)/k` | `smooth_lower_bound()` |
| STE clamp | 前向=clamp, 反向=恒等 | `_StraightThroughClamp` |
| 角度归一化 | `atan2(sin(θ), cos(θ))` | `normalize_angle()` |
| 速度下界 | `softplus(v, beta=10)` ≈ `max(v, 0)` | `vehicle.py:33` |

### 8.4 PyTorch API 在本项目中的用法

| API | 用途 | 项目中的位置 |
|-----|------|-------------|
| `nn.Module` | 可训练模块基类 | `LatControllerTruck`, `LonController`, `DiffControllerParams` |
| `nn.Parameter(tensor)` | 声明可训练参数 | `lat_truck.py:62`, `lon.py:33-39` |
| `register_buffer(name, tensor)` | 注册不可训练常量 | `lat_truck.py:36,58,60` |
| `tensor.detach()` | 切断计算图连接 | `vehicle.py:39`, `trajectory.py:155` |
| `tensor.item()` | tensor → Python float | V1 路径中大量使用 |
| `torch.no_grad()` | 禁止构建计算图 | `trajectory.py:162` |
| `loss.backward()` | 反向传播 | `train.py:227` |
| `optimizer.zero_grad()` | 清零累积梯度 | `train.py:196` |
| `optimizer.step()` | 用梯度更新参数 | `train.py:238` |
| `clip_grad_norm_()` | 梯度范数裁剪 | `train.py:236` |
| `torch.autograd.Function` | 自定义前向/反向 | `common.py:136` |
| `param.register_hook(fn)` | 梯度钩子（反向时触发） | `train.py:173-174` |
| `model.parameters()` | 获取所有可训练参数 | `train.py:179` |
| `model.named_parameters()` | 带名字的参数迭代器 | `train.py:179` |
| `torch.stack()` | 堆叠 tensor 列表 | `train.py:91-95` |
| `torch.searchsorted()` | 排序数组中二分查找 | `common.py:18` |
