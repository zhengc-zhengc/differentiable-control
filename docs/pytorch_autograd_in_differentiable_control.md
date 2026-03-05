# PyTorch 自动微分与可微控制器：从原理到实现

> 本文以本项目的可微控制器为载体，讲解 PyTorch 自动微分的核心概念。
> 每个知识点都对应项目中的真实代码，标注了文件路径和行号。

---

## 目录

1. [基础概念：Tensor 与计算图](#1-基础概念tensor-与计算图)
2. [自动微分的核心规则](#2-自动微分的核心规则)
3. [nn.Module 与 nn.Parameter：可训练参数的管理](#3-nnmodule-与-nnparameter可训练参数的管理)
4. [项目全局：闭环仿真的计算图](#4-项目全局闭环仿真的计算图)
5. [逐行解读：横向控制器的可微路径](#5-逐行解读横向控制器的可微路径)
6. [逐行解读：纵向控制器的可微路径](#6-逐行解读纵向控制器的可微路径)
7. [不可微操作与替代方案](#7-不可微操作与替代方案)
8. [梯度爆炸：原理与防治](#8-梯度爆炸原理与防治)
9. [训练 Pipeline：从 loss 到参数更新](#9-训练-pipeline从-loss-到参数更新)
10. [常见陷阱速查表](#10-常见陷阱速查表)

---

## 1. 基础概念：Tensor 与计算图

### 1.1 什么是 Tensor

Tensor 是 PyTorch 的核心数据结构，可以理解为"带额外功能的数组"。与 numpy 数组不同的是，**tensor 会记住自己是怎么被计算出来的**。

```python
import torch

a = torch.tensor(3.0)    # 标量 tensor
b = torch.tensor(2.0)
c = a * b + 1.0          # c = 7.0, 且 c 记住了"我是 a*b+1 得到的"
```

### 1.2 计算图（Computation Graph）

当你用 tensor 做运算时，PyTorch 在后台构建一棵**计算图**——记录每个值是由哪些运算、哪些输入产生的。

```
a (3.0) ──→ [×] ──→ tmp (6.0) ──→ [+1] ──→ c (7.0)
b (2.0) ──↗
```

调用 `c.backward()` 时，PyTorch 沿这棵图反向遍历，用链式法则自动算出 dc/da 和 dc/db。

**这就是"自动微分"的全部核心**：只要你的计算过程使用 tensor 运算，PyTorch 就能自动算梯度。不需要你手动推导任何导数公式。

### 1.3 计算图在项目中的体现

在本项目中，一次 50Hz 闭环仿真就是一棵巨大的计算图：

```
车辆状态(x,y,yaw,v)   ←─ tensor
      ↓
横向控制器(查表、PID)   ←─ 中间运算全是 tensor
      ↓
steer_out              ←─ tensor
      ↓
纵向控制器(PID)        ←─ 中间运算全是 tensor
      ↓
acc_cmd                ←─ tensor
      ↓
vehicle.step()         ←─ 运动学方程，tensor 运算
      ↓
新的车辆状态           ←─ tensor
      ↓
[重复 N 步]
      ↓
loss = Σ(误差²)        ←─ 标量 tensor
```

`loss.backward()` 一调用，梯度就从 loss 沿着这棵图一路回传到控制器参数（T2_y, station_kp 等）。

---

## 2. 自动微分的核心规则

### 2.1 规则一：tensor 运算 → 可微

PyTorch 内置的 tensor 运算（加减乘除、三角函数、指数对数等）都已经预注册了导数规则。你用它们计算，梯度就能自动传播。

**项目实例 — 车辆运动学方程** (`sim/model/vehicle.py:28-31`)：

```python
# 每一步运算都是 tensor 操作，PyTorch 知道每个函数的导数：
# d(cos)/d(yaw) = -sin(yaw),  d(sin)/d(yaw) = cos(yaw),  d(tan)/d(delta) = 1/cos²(delta)

self.x = self.x + self.v * torch.cos(self.yaw) * self.dt    # x_new = x + v·cos(θ)·dt
self.y = self.y + self.v * torch.sin(self.yaw) * self.dt    # y_new = y + v·sin(θ)·dt
self.yaw = self.yaw + self.v * torch.tan(delta) / self.L * self.dt  # θ_new = θ + v·tan(δ)/L·dt
self.v = self.v + acc * self.dt                              # v_new = v + a·dt
```

这四行代码包含了 `cos`, `sin`, `tan`, 加法, 乘法, 除法——全部是 PyTorch 内置可微运算。
所以**车辆状态对控制输入 (delta, acc) 的梯度自动可得**。

### 2.2 规则二：`.item()` → 断链

`.item()` 把 tensor 变成 Python float。float 不是 tensor，不在计算图内，梯度到此为止。

**项目实例 — V1 路径** (`sim/controller/lat_truck.py:134-141`)：

```python
# V1 路径：每个 lookup1d 的结果都 .item() 了
prev_time_dist = lookup1d(self.T2_x, self.T2_y, speed_t).item()   # tensor → float，梯度断了
reach_time_theta = lookup1d(self.T3_x, self.T3_y, speed_t).item() # 同上
```

**对比 — differentiable 路径** (`sim/controller/lat_truck.py:233-234`)：

```python
# differentiable 路径：不调 .item()，结果仍是 tensor，梯度保持连通
prev_time_dist = lookup1d(self.T2_x, self.T2_y, speed_kph_safe)   # 返回 tensor
reach_time_theta = lookup1d(self.T3_x, self.T3_y, speed_kph_safe) # 返回 tensor
```

**总结**：V1 路径在每一步都调 `.item()` 把 tensor 变成 float，整个计算链变成了普通 Python 浮点运算，不可微。
differentiable 路径不调 `.item()`，tensor 一路贯穿，计算图完整。

### 2.3 规则三：`.detach()` → 主动断链

`.detach()` 返回一个新 tensor，值相同但**不在计算图内**。常用于主动截断梯度。

**项目实例 — 控制器状态更新** (`sim/controller/lat_truck.py:293`)：

```python
self.steer_fb_prev = steer_fb.detach().clone()
# detach(): 从计算图中脱离（下一步用这个值时，梯度不会穿过它）
# clone(): 复制一份（防止后续 in-place 修改影响当前计算图）
```

为什么要 detach？因为 `steer_fb_prev` 是"上一步的转向角"，如果不 detach，
梯度就会通过 `prev → rate_limit → steer_fb` 一直回溯到很早的时间步，
导致计算图越来越长、显存爆炸。detach 就是在说："下一步用的这个值，当作一个常数，不追溯它的来历。"

### 2.4 规则四：Python `if/else` on tensor → 不可微

Python 的 `if` 语句接受 bool 值，不接受 tensor。当你写 `if tensor_value > threshold:` 时，
实际上先做了隐式 `.item()` 转换。更重要的是，**分支选择本身不可微**——
要么走 if，要么走 else，没有中间状态。

**项目实例 — V1 路径离散分支** (`sim/controller/lon.py:173-178`)：

```python
# V1：硬切换，switch_speed 不在计算图中
if speed_mps <= self.switch_speed.item():    # .item() 取出 float，离散判断
    kp = self.low_speed_kp.item()            # 选择低速增益（float）
else:
    kp = self.high_speed_kp.item()           # 选择高速增益（float）
```

**对比 — differentiable 路径用 smooth_step 替代** (`sim/controller/lon.py:308-310`)：

```python
# differentiable：sigmoid 连续混合，所有参数都能收到梯度
w_low = 1.0 - smooth_step(speed_mps, self.switch_speed, temp=0.5)
#              ↑ sigmoid((speed - switch_speed) / 0.5)
#              speed << switch_speed → w_low ≈ 1（全低速）
#              speed >> switch_speed → w_low ≈ 0（全高速）
#              speed ≈ switch_speed  → w_low ≈ 0.5（混合）

kp = w_low * self.low_speed_kp + (1.0 - w_low) * self.high_speed_kp
#    全部是 tensor 乘法加法，梯度通畅
```

---

## 3. nn.Module 与 nn.Parameter：可训练参数的管理

### 3.1 nn.Parameter vs register_buffer

PyTorch 用两种方式在 Module 中存储数据：

| 类型 | 是否参与梯度优化 | 用途 |
|------|----------------|------|
| `nn.Parameter` | **是** | 要学习的参数 |
| `register_buffer` | **否** | 固定常量（参与模型保存但不参与优化） |

**项目实例 — 横向控制器** (`sim/controller/lat_truck.py:36, 49-62`)：

```python
# Buffer：物理参数，不优化（优化了也没物理意义）
self.register_buffer('kLh', torch.tensor(float(lat['kLh'])))  # 铰接修正系数

# 8 张查找表的处理
_fixed_tables = {'T1', 'T7', 'T8'}   # 安全/物理约束表
for name in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
    xs, ys = table_from_config(lat[key_map[name]])
    self.register_buffer(f'{name}_x', xs)          # x 轴（速度断点）永远固定
    if name in _fixed_tables:
        self.register_buffer(f'{name}_y', ys)       # T1/T7/T8 y 值固定：最大误差角/最大转向角/侧滑参数
    else:
        setattr(self, f'{name}_y', nn.Parameter(ys)) # T2-T6 y 值可学习：预瞄距离/收敛时间/...
```

**项目实例 — 纵向控制器** (`sim/controller/lon.py:33-55`)：

```python
# nn.Parameter：7 个可优化标量
self.station_kp = nn.Parameter(torch.tensor(float(lon['station_kp'])))   # 站位 PID 比例增益
self.station_ki = nn.Parameter(torch.tensor(float(lon['station_ki'])))   # 站位 PID 积分增益
self.low_speed_kp = nn.Parameter(torch.tensor(float(lon['low_speed_kp'])))
self.low_speed_ki = nn.Parameter(torch.tensor(float(lon['low_speed_ki'])))
self.high_speed_kp = nn.Parameter(torch.tensor(float(lon['high_speed_kp'])))
self.high_speed_ki = nn.Parameter(torch.tensor(float(lon['high_speed_ki'])))
self.switch_speed = nn.Parameter(torch.tensor(float(lon['switch_speed'])))  # 低/高速切换点

# register_buffer：5 张查找表 y 值（物理限制，不优化）
for name, key in [('L1', 'L1_acc_up_lim'), ('L2', 'L2_acc_low_lim'), ...]:
    self.register_buffer(f'{name}_y', ys)   # 加速度上下限、速率限制——物理约束
```

### 3.2 梯度是怎么到达 nn.Parameter 的

nn.Parameter 本质就是一个 `requires_grad=True` 的 tensor。
当它参与运算时，结果 tensor 会记住自己依赖这个 Parameter。
`loss.backward()` 时，梯度沿计算图回传，最终累积到 `Parameter.grad` 上。

**以 `station_kp` 为例追踪完整链路：**

```
loss                                      ← tracking_loss() 计算的标量
 ↑
acc_cmd                                   ← lon_ctrl 返回的 tensor
 ↑
acc_closeloop = kp * speed_input + ...    ← PID 公式 (lon.py:315-318)
 ↑                                          kp 是从 station_kp 计算来的
kp = w_low * low_speed_kp + ...           ← smooth_step 混合 (lon.py:309)
 ↑
self.low_speed_kp                         ← nn.Parameter，梯度最终到达这里
```

---

## 4. 项目全局：闭环仿真的计算图

### 4.1 一个时间步的数据流

`sim/sim_loop.py:82-132` (differentiable=True 路径)：

```python
# ① 误差计算：车辆位置(tensor) vs 参考轨迹(float) → 误差(tensor)
ref_pt = analyzer.query_nearest_by_position(car.x, car.y)  # 返回 float TrajectoryPoint
lateral_error = torch.cos(ref_theta) * dy - torch.sin(ref_theta) * dx  # tensor 运算
heading_error = normalize_angle(car.yaw - ref_theta)                    # tensor 运算

# ② 横向控制器：误差(tensor) + 参数(nn.Parameter) → 转向(tensor)
steer_out, kappa_cur, kappa_near, curvature_far = lat_ctrl.compute(
    x=car.x, y=car.y, yaw_deg=car.yaw_deg, speed_kph=car.speed_kph,
    yawrate=yawrate, steer_feedback=prev_steer, analyzer=analyzer, ...)

# ③ 纵向控制器：速度(tensor) + 参数(nn.Parameter) → 加速度(tensor)
acc_cmd = lon_ctrl.compute(
    x=car.x, y=car.y, yaw_deg=car.yaw_deg, speed_kph=car.speed_kph,
    curvature_far=curvature_far, analyzer=analyzer, t_now=t, ...)

# ④ 存入历史（tensor 引用，不做 .item()）
history.append({
    'lateral_error': lateral_error,  # tensor：loss 直接从这里计算
    'heading_error': heading_error,  # tensor
    'v': car.v, 'steer': steer_out, 'acc': acc_cmd,  # 全部 tensor
})

# ⑤ 车辆更新：tensor 运动学方程
car.step(delta=steer_out / steer_ratio * DEG2RAD, acc=acc_cmd)
```

### 4.2 跨时间步的梯度传播

车辆状态在每一步更新后传入下一步，形成**时间上的链式依赖**：

```
step 0:  state₀ → ctrl₀ → steer₀/acc₀ → state₁
step 1:  state₁ → ctrl₁ → steer₁/acc₁ → state₂
step 2:  state₂ → ctrl₂ → steer₂/acc₂ → state₃
...
```

`loss.backward()` 时，梯度从 loss 回传到每一步的 lateral_error，
再通过 state 的依赖关系一直传到 step 0 的控制器参数。
这就是 **BPTT（Backpropagation Through Time）**——反向传播穿越时间。

**Truncated BPTT** (`sim/sim_loop.py:84-86`)：

```python
# 每 K 步截断一次：防止梯度链太长导致爆炸
if tbptt_k > 0 and step > 0 and step % tbptt_k == 0:
    car.detach_state()             # 车辆状态从计算图中脱离
    prev_steer = prev_steer.detach()  # 上一步转向也脱离

# detach_state() 的实现 (sim/model/vehicle.py:37-42):
def detach_state(self):
    self.x = self.x.detach().requires_grad_(False)     # 当作"新的起点"
    self.y = self.y.detach().requires_grad_(False)
    self.yaw = self.yaw.detach().requires_grad_(False)
    self.v = self.v.detach().requires_grad_(False)
```

截断后，step K 之后的梯度无法回传到 step K 之前。相当于每 K 步"重新开始"一段计算图。

---

## 5. 逐行解读：横向控制器的可微路径

> 文件：`sim/controller/lat_truck.py`，方法 `_compute_differentiable` (L202-310)

### Step 1: 查表 (L231-239)

```python
max_theta_deg = lookup1d(self.T1_x, self.T1_y, speed_kph_safe)
#               T1_y 是 buffer → 返回的 tensor 不连接到任何 nn.Parameter
#               用于限制 target_theta，是安全约束

prev_time_dist = lookup1d(self.T2_x, self.T2_y, speed_kph_safe)
#                T2_y 是 nn.Parameter ← 梯度从这里开始
#                含义：预瞄距离 = speed × prev_time_dist

near_pt_time = lookup1d(self.T5_x, self.T5_y, speed_kph_safe)
#              T5_y 是 nn.Parameter，含义：近预瞄时间偏移
far_pt_time = lookup1d(self.T6_x, self.T6_y, speed_kph_safe)
#             T6_y 是 nn.Parameter，含义：远预瞄时间偏移
```

`lookup1d` 本身如何可微？见 `sim/common.py:7-25`：

```python
def lookup1d(table_x, table_y, x):
    # table_x 的值用 .item() 提取为 float（断点位置是固定的，不需要梯度）
    x_clamped = torch.clamp(x, table_x[0].item(), table_x[-1].item())

    # searchsorted 找到 x 落在哪个区间（离散索引操作，不可微）
    idx = torch.searchsorted(table_x, x_clamped) - 1

    # 但关键的可微部分在这里——线性插值：
    y0, y1 = table_y[i], table_y[i + 1]   # table_y 可能是 nn.Parameter
    t = (x_clamped - x0) / (x1 - x0)       # 插值权重（tensor）
    return y0 + (y1 - y0) * t               # 结果对 y0, y1 可微！
    #      ↑ 如果 table_y 是 nn.Parameter，梯度通过这里回传
```

### Step 2: 轨迹查询 (L241-248)

```python
currt = analyzer.query_nearest_by_position(x, y)
#       内部用 detach + no_grad + argmin → 返回 float TrajectoryPoint
#       "当前最近的参考点在哪"是离散的选择，不可微——这是设计选择

t_base = torch.tensor(currt.t)             # 当前时间（float → 新的叶子 tensor）
near_kappa, _, _, _ = analyzer.query_by_time_differentiable(
    t_base + near_pt_time)                  # near_pt_time 是 tensor（来自 T5_y）
#   ↑ query_by_time_differentiable 内部用 lookup1d 做可微插值
#   ↑ 梯度链：T5_y → near_pt_time → t_base + near_pt_time → lookup1d → near_kappa
```

`query_by_time_differentiable` 的实现 (`sim/model/trajectory.py:198-207`)：

```python
def query_by_time_differentiable(self, t: torch.Tensor):
    """用 lookup1d 对轨迹的各分量按时间插值，返回 tensor，对 t 可微。"""
    kappa = lookup1d(self._t_arr, self._kappa_arr, t)  # _kappa_arr 是 buffer（轨迹数据）
    v = lookup1d(self._t_arr, self._v_arr, t)          # 轨迹数据不需要梯度
    a = lookup1d(self._t_arr, self._a_arr, t)          # 但 t 是 tensor，梯度通过 t 回传
    s = lookup1d(self._t_arr, self._s_arr, t)
    return kappa, v, a, s
    # 梯度流向：loss → kappa → lookup1d 的插值权重 → t → near_pt_time → T5_y
```

### Step 3-7: 控制律（全 tensor 运算，天然可微） (L250-286)

```python
# 误差计算
lateral_error = cos_theta * dy - sin_theta * dx    # cos/sin 是 float（currt 的），dy/dx 是 tensor
heading_error = normalize_angle(yaw_rad - currt.theta)  # atan2(sin, cos)，天然可微

# Step 6 关键：target_theta 的限幅
error_angle_raw = torch.atan(dis2lane / prev_dist)       # atan 可微
target_theta = _straight_through_clamp(
    error_angle_raw, -max_err_angle_limit, max_err_angle_limit)
#  ↑ STE：前向硬限幅，反向梯度 = 1（见第 7 节详解）

# Step 7：目标曲率
target_curvature = ((target_theta - real_theta)
                    + (target_dt_theta - real_dt_theta) * T_dt) / denom
#                    全部是 tensor 加减乘除，自动可微
```

### Step 8-10: 输出限幅（STE + smooth_clamp） (L288-310)

```python
# 反馈转向角
steer_fb = rate_limit(self.steer_fb_prev, steer_fb_raw,
                      self.rate_limit_fb, dt, differentiable=True)
#          ↑ differentiable=True 时内部用 STE clamp
self.steer_fb_prev = steer_fb.detach().clone()
#                    ↑ detach：截断跨步梯度，只保留当步内的梯度

# 最终合并
steer_raw = smooth_clamp(steer_fb + steer_ff,
                         -max_steer_angle, max_steer_angle, temp=1.0)
#           ↑ tanh 平滑限幅：梯度在边界处不为零（但 temp=1.0 时梯度 ≤ 1，安全）
```

---

## 6. 逐行解读：纵向控制器的可微路径

> 文件：`sim/controller/lon.py`，方法 `_compute_differentiable` (L233-372)

### Step 1: Frenet 变换 (L259-275)

```python
# 修复后：直接传入 tensor，不再 .item()
s_match, s_dot, d_frenet, d_dot = analyzer.to_frenet(
    x, y, yaw_rad, speed_mps, matched)
#   ↑ to_frenet 内部全是 tensor 运算（sin, cos, 加减），天然可微
#   ↑ matched 是 float TrajectoryPoint（当作常数参考点）
#   ↑ x, y, yaw_rad, speed_mps 是 tensor → s_match 对它们可微

# 可微时间查询
_, _, _, ref_s = analyzer.query_by_time_differentiable(t_now_t)
_, spd_v, _, _ = analyzer.query_by_time_differentiable(
    t_now_t + self.preview_window * dt)

station_error = ref_s - s_match          # 两个 tensor 相减，可微
preview_speed_error = spd_v - speed_mps  # spd_v(轨迹插值) - speed_mps(车辆状态)
```

### Step 3-4: PID 与增益混合 (L299-319)

```python
# 站位 PID：传入 nn.Parameter 作为增益
speed_offset = self.station_pid.control(
    station_fnl, dt,
    kp=self.station_kp,   # nn.Parameter，直接传入（不 .item()）
    ki=self.station_ki,   # nn.Parameter
    kd=0.0,
    integrator_enable=self.station_integrator_enable,
    sat=self.station_sat, differentiable=True)
```

PID 内部的梯度流 (`sim/common.py:189-200`)：

```python
def control(self, error, dt, kp, ki, kd, integrator_enable, sat, differentiable=False):
    if integrator_enable:
        self.integral = clamp(self.integral + error * dt, -sat, sat, differentiable)
        #               ↑ differentiable=True 时用 smooth_clamp（梯度不消失）
    derivative = (error - self.prev_error) / dt
    self.prev_error = error.detach().clone()
    #               ↑ detach：上一步的误差不参与当步的梯度

    return kp * error + ki * self.integral + kd * derivative
    #      ↑              ↑
    #      kp 是 nn.Parameter, error 是 tensor → kp 的梯度 = d(loss)/d(output) × error
    #      ki 是 nn.Parameter, integral 是 tensor → ki 的梯度 = d(loss)/d(output) × integral
```

**为什么 PID 增益能收到梯度？** 因为 `kp * error` 中 kp 是 nn.Parameter，
它和 error 做乘法时进入了计算图。`d(output)/d(kp) = error`，
所以 kp 的梯度 = `d(loss)/d(output) × error`。
即使 error 的来源（轨迹查询）部分不可微，kp 作为乘法系数仍能获得有意义的梯度。

### Step 6: 大量 smooth_step 替代 if/else (L337-359)

```python
# 曲率条件：V1 是 if curvature < -0.0075: acc_up *= 0.75
# differentiable 用 sigmoid 混合
w_curv = smooth_step(-curvature_far, 0.0075, temp=0.01)
#        sigmoid((-curvature_far - 0.0075) / 0.01)
#        curvature_far < -0.0075 → w_curv ≈ 1 → acc_up_lim *= 0.75
#        curvature_far > -0.0075 → w_curv ≈ 0 → acc_up_lim 不变
acc_up_lim_adj = acc_up_lim * (1.0 - 0.25 * w_curv)  # tensor 运算，可微

# 低速蠕行保护：V1 是 if abs(speed) >= 0.2 or acc >= 0.25: pass; else: min(-0.05, acc)
# differentiable 用 3 个 smooth_step 叠加
w_pass = smooth_step(abs_speed, 0.2, temp=0.05)     # "速度够大"的 sigmoid 权重
w_acc_ok = smooth_step(acc_clamped, 0.25, temp=0.05) # "加速度够大"的 sigmoid 权重
w_normal = 1.0 - (1.0 - w_pass) * (1.0 - w_acc_ok)  # 相当于 OR 逻辑的可微版本
#          两个 sigmoid 相乘 = soft AND，1 - soft AND = soft OR
```

---

## 7. 不可微操作与替代方案

### 7.1 硬限幅 (clamp) → Straight-Through Estimator

**问题**：`torch.clamp(x, lo, hi)` 在 x 饱和时梯度为零。
在控制器中，rate limiter 经常饱和（比如转向速率到顶），导致大量时间步梯度为零。

**解决**：STE (`sim/common.py:136-158`)

```python
class _StraightThroughClamp(torch.autograd.Function):
    """自定义 autograd 函数：手动定义前向和反向行为。"""

    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x, lo, hi)       # 保存输入（反向时要用）
        return torch.clamp(x, lo.item(), hi.item())  # 前向：正常硬限幅

    @staticmethod
    def backward(ctx, grad_output):
        x, lo, hi = ctx.saved_tensors
        grad_x = grad_output   # ← 关键：无条件把梯度传给 x（即使 x 被 clamp 了）
        # 对比普通 clamp：x 饱和时 grad_x = 0
        return grad_x, ...
```

**`torch.autograd.Function` 是什么？**
当 PyTorch 内置运算不满足需求时，你可以自定义一个运算的前向和反向行为。
只需实现 `forward()` 和 `backward()` 两个方法。
这是 PyTorch 自动微分的"逃生通道"——让你精确控制梯度该怎么传。

### 7.2 阶跃/符号函数 → smooth 近似

**问题**：`sign(x)` 在 x=0 处导数未定义，在其他地方导数为零。
`step(x > thr)` 也是阶跃函数，不可微。

**解决**：用 `tanh` 或 `sigmoid` 做平滑近似 (`sim/common.py:74-86`)

```python
def smooth_sign(x, temp=0.01):
    return torch.tanh(x / temp)      # temp 越小越接近 sign，但导数越大
    # temp=0.01 时 x=0 处导数 = 1/0.01 = 100 ← 潜在的梯度爆炸源！
    # temp=0.5  时 x=0 处导数 = 1/0.5  = 2   ← 安全

def smooth_step(x, threshold, temp=1.0):
    return torch.sigmoid((x - threshold) / temp)  # 平滑阶跃
    # temp=1.0 时最大导数 = 0.25 ← 安全
    # temp=0.01 时最大导数 = 25  ← 需要注意
```

**关于 temperature 参数**：

temp 越小，smooth 函数越接近原始硬函数（近似更精确），
但过渡区的导数越大（梯度放大越严重）。
这是一个 **精度 vs 梯度稳定性** 的权衡。

### 7.3 速度下限 → softplus

**问题**：`clamp(v, min=0)` 在 v=0 处梯度为零。车减速到零后，加速度的梯度传不回去。

**解决** (`sim/model/vehicle.py:32-33`)：

```python
if self.differentiable:
    self.v = torch.nn.functional.softplus(self.v, beta=10.0)
    # softplus(x) = log(1 + exp(β·x)) / β
    # x > 0 时 ≈ x（原值不变）
    # x ≈ 0 时 ≈ log(2)/β ≈ 0.07（不会真正为零）
    # 导数永远 > 0：sigmoid(β·x)
else:
    self.v = torch.clamp(self.v, min=0.0)
    # 硬限幅：v<0 时梯度为零
```

### 7.4 下界/上界约束 → softplus

**问题**：`max(x, lo)` 在 x<lo 时导数为零。

**解决** (`sim/common.py:47-55`)：

```python
def smooth_lower_bound(x, lo, sharpness=10.0):
    """smooth_max(x, lo)：保证输出 ≥ lo，且导数不为零。"""
    return lo + softplus((x - lo) * sharpness) / sharpness
    # x >> lo 时 ≈ x（保持原值）
    # x << lo 时 ≈ lo（但有微小的正导数）
    # 导数永远 ∈ (0, 1]
```

**使用位置** (`sim/controller/lat_truck.py:269-270`)：

```python
prev_dist = smooth_lower_bound(speed_mps * prev_time_dist, self.min_prev_dist)
#           保证预瞄距离 ≥ 5.0m，即使速度很低也不会除以零
```

### 7.5 总结：不可微操作替代方案表

| 原始操作 | 问题 | 可微替代 | 项目中的位置 |
|---------|------|---------|------------|
| `clamp(x, lo, hi)` 饱和区 | 梯度=0 | STE: 前向 clamp, 反向梯度=1 | rate_limit, target_theta |
| `sign(x)` | 处处梯度=0 | `tanh(x/temp)` | 已被 STE clamp 替代 |
| `if x > thr:` | Python 分支不可微 | `sigmoid((x-thr)/temp)` 连续混合 | 低/高速切换, 曲率条件 |
| `max(x, lo)` | x<lo 时梯度=0 | `lo + softplus(...)` | 预瞄距离下限 |
| `min(x, hi)` | x>hi 时梯度=0 | `hi - softplus(...)` | 低速蠕行限制 |
| `clamp(v, min=0)` | v<0 时梯度=0 | `softplus(v, beta)` | 车辆速度下限 |

---

## 8. 梯度爆炸：原理与防治

### 8.1 为什么闭环控制容易梯度爆炸

BPTT 的梯度是沿时间步链式相乘的：

```
d(loss)/d(θ) = Σ_t  d(loss)/d(state_t) × ∏_{k=t}^{T} d(state_{k+1})/d(state_k) × d(state_t)/d(θ)
                                           ↑ 这个连乘是关键
```

如果每步的局部 Jacobian `d(state_{k+1})/d(state_k)` 的谱半径 > 1，
N 步连乘就指数增长。

### 8.2 具体案例：smooth_sign 的 temp 过小

**项目中的真实事故**（修复前）：

横向控制器 Step 6 原来用 `smooth_sign(error, temp=0.01)` 做符号函数的平滑近似。
`tanh(x/0.01)` 在 x=0 附近的导数 = 1/0.01 = **100**。

每个时间步，误差信号经过 `smooth_sign` 时被放大 100 倍。
BPTT 链式乘法跨 32 步：100^32 = 10^64 → **Inf**。

但 loss 本身完全正常（~65），因为 forward 时 smooth_sign 大多数时间在饱和区（导数≈0），
只在信号过零的瞬间导数很大。forward 稳定，backward 爆炸。

**有限差分验证**（证明真实梯度很小）：

```
参数: high_speed_kp = 0.34
有限差分梯度:  -19.87     ← (f(x+ε)-f(x-ε))/(2ε)，真实梯度
BPTT 梯度:     NaN        ← 链式乘法数值溢出
```

### 8.3 根治方案：化简数学表达式

用户（你）发现了关键简化：C++ 原始代码的 `sign(x) * min(|x|, L)` 数学上等价于 `clamp(x, -L, L)`。
`clamp` 的导数在范围内恒为 1（无放大），用 STE clamp 后梯度完全稳定。

修复前后对比 (`sim/controller/lat_truck.py` Step 6)：

```python
# 修复前（逐步翻译 C++ 的分解形式）：
target_theta = smooth_sign(error_angle_raw, temp=0.01) * max_err_angle
#              导数 100 × N 步 → 爆炸

# 修复后（化简后的等价形式）：
target_theta = _straight_through_clamp(
    error_angle_raw, -max_err_angle_limit, max_err_angle_limit)
#              导数恒为 1 → 稳定
```

**教训**：将硬限幅代码改为可微版本时，**先化简数学表达式再选择近似方式**。
不必忠实翻译 C++ 的每一步分解。

### 8.4 保护层：三层安全网

即使根因修复了，仍保留三层保护防止意外：

**层 1: TBPTT** (`sim/sim_loop.py:84-86`)

```python
if tbptt_k > 0 and step > 0 and step % tbptt_k == 0:
    car.detach_state()         # 每 64 步截断
```

效果：梯度链最长只有 64 步（1.28 秒），而非整段仿真。

**层 2: 参数梯度 Hook** (`sim/optim/train.py:166-170`)

```python
def _sanitize_grad(grad):
    g = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)  # NaN/Inf → 0
    return g.clamp(-1e4, 1e4)   # 元素级限幅
```

效果：即使某个参数梯度爆炸了，也不会污染 Adam 优化器的动量。

**层 3: 全局梯度范数裁剪** (`sim/optim/train.py:236-237`)

```python
grad_norm = torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=10.0)
# 如果所有参数梯度的 L2 范数 > 10，按比例缩小
```

效果：防止单个 batch/轨迹的异常梯度导致参数跳太远。

---

## 9. 训练 Pipeline：从 loss 到参数更新

> 文件：`sim/optim/train.py`

### 9.1 Loss 的组成 (L74-125)

```python
def tracking_loss(history, ref_speed, ...):
    lat_errs = torch.stack([h['lateral_error'] for h in history])   # 横向误差序列
    head_errs = torch.stack([h['heading_error'] for h in history])  # 航向误差序列
    speeds = torch.stack([h['v'] for h in history])                 # 速度序列
    steers = torch.stack([h['steer'] for h in history])             # 转向序列

    loss = 10.0 * (lat_errs ** 2).mean()       # 横向误差 MSE（权重最大）
         +  5.0 * (head_errs ** 2).mean()       # 航向误差 MSE
         +  1.0 * (speed_errs ** 2).mean()      # 速度误差 MSE
         + 0.01 * (steer_rate ** 2).mean()       # 转向平滑度惩罚
         + 0.01 * (acc_rate ** 2).mean()         # 加速度平滑度惩罚
```

所有这些量都是 tensor，`loss.backward()` 时梯度沿计算图回传。

### 9.2 一次 epoch 的流程 (L194-238)

```python
for epoch in range(n_epochs):
    optimizer.zero_grad()              # ① 清空上一轮的梯度

    for traj_name in trajectories:     # ② 多轨迹训练（circle, sine, combined）
        history = run_simulation(      # ③ 前向：跑一次 50Hz 闭环仿真
            traj, differentiable=True, tbptt_k=64)
        loss, details = tracking_loss(history, ref_speed)  # ④ 计算 loss
        epoch_loss += loss             # ⑤ 累积多轨迹的 loss

    epoch_loss.backward()              # ⑥ 反向传播：计算所有参数的梯度
    # 此时每个 nn.Parameter 的 .grad 属性被填充

    clip_grad_norm_(max_norm=10.0)     # ⑦ 全局梯度裁剪
    optimizer.step()                   # ⑧ Adam 更新参数
```

### 9.3 参数分组与学习率 (L176-188)

```python
# 查找表 y 值 用较低学习率（它们是多点值，变化敏感）
# PID 增益标量 用较高学习率（单个标量，需要更快收敛）
optimizer = torch.optim.Adam([
    {'params': other_params, 'lr': 1e-3},      # station_kp, switch_speed 等
    {'params': table_params, 'lr': 5e-4},       # T2_y, T3_y 等查找表
])
```

---

## 10. 常见陷阱速查表

| 陷阱 | 症状 | 原因 | 本项目中的实例 |
|------|------|------|--------------|
| `.item()` 断链 | 参数梯度为 None 或 0 | tensor → float，脱离计算图 | V1 路径所有 lookup1d 后的 .item() |
| `.detach()` 遗忘 | 显存持续增长 / 梯度爆炸 | 内部状态未从计算图脱离 | PID.prev_error, steer_fb_prev 需要 detach |
| smooth temp 过小 | 梯度 NaN/Inf | 过渡区导数 = 1/temp，链式乘法指数增长 | smooth_sign(temp=0.01) → 导数 100 |
| Python if on tensor | 参数无梯度 | 分支选择不在计算图内 | V1 路径的 speed > switch_speed |
| hard clamp 饱和 | 参数不更新 | 饱和区梯度为零 | rate_limit 不加 STE 时 |
| 死参数 | 参数有梯度但永远为零 | 输出不参与 loss 计算 | T5_y 的 near_kappa 不参与控制 |
| 积分器关闭 | ki 梯度为零 | 积分项 = 0 → ki × 0 = 0 | station_ki (integrator_enable=false) |

---

## 附录：关键文件索引

| 文件 | 内容 | 自动微分相关要点 |
|------|------|----------------|
| `sim/common.py` | 基础运算 | lookup1d, smooth_*, STE, PID, IIR |
| `sim/model/vehicle.py` | 自行车模型 | 运动学方程全 tensor; softplus 速度下限 |
| `sim/model/trajectory.py` | 轨迹生成+分析 | query_by_time_differentiable 可微; query_by_position 不可微(detach) |
| `sim/controller/lat_truck.py` | 横向控制器 | _compute_v1 (float) vs _compute_differentiable (tensor) |
| `sim/controller/lon.py` | 纵向控制器 | smooth_step 替代 if/else; STE 替代 clamp |
| `sim/sim_loop.py` | 仿真主循环 | TBPTT 截断; tensor/float 双路径 |
| `sim/optim/train.py` | 训练流程 | loss 组成; 梯度 hook; clip_grad_norm_; Adam 分组 |
