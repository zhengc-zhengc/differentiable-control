# 反向传播如何穿过控制闭环

> 本文以本项目的可微控制器为例，从零解释"梯度是如何从 loss 传回到控制器参数"的全过程。
> 假设读者了解基本的导数概念（如 `y = 2x` 则 `dy/dx = 2`），但不要求深度学习背景。

---

## 1. 我们的目标

我们有一个**控制器**，它读取车辆当前状态（位置、速度、航向），输出方向盘转角和加速度指令。
控制器内部有很多**参数**（查找表的 Y 值、PID 增益），这些参数决定了控制器的行为。

我们希望找到一组参数，使车辆尽可能精确地跟踪参考轨迹。

**问题**：如何知道"把某个参数调大一点，跟踪效果会变好还是变差"？

**答案**：计算 loss 对该参数的**梯度**（导数）。梯度告诉我们调参的方向和幅度。

---

## 2. 什么是计算图

PyTorch 的核心机制：**每次对 tensor 做运算，都会自动记录这步运算**。

一个简单的例子：

```python
a = torch.tensor(3.0, requires_grad=True)   # 可调参数
b = a * 2                                    # PyTorch 记录："b 来自 a × 2"
c = b + 1                                    # PyTorch 记录："c 来自 b + 1"
loss = c ** 2                                # PyTorch 记录："loss 来自 c²"
```

内存中形成了一条链：

```
a ──×2──→ b ──+1──→ c ──²──→ loss
```

这就是**计算图**：每个节点是一个 tensor，每条边是一步运算。

---

## 3. 反向传播 = 沿计算图逆向求导

调用 `loss.backward()` 时，PyTorch 从 loss 出发，**逆着箭头**逐步应用链式法则：

```
loss = c²         →  d(loss)/dc = 2c = 2×8 = 16
c = b + 1         →  dc/db = 1
b = a × 2         →  db/da = 2

链式法则：d(loss)/da = d(loss)/dc × dc/db × db/da
                     = 16 × 1 × 2
                     = 32
```

**关键理解**：PyTorch 不需要你写出这些公式。你只管 forward（前向计算），它自动帮你 backward（反向求导）。

---

## 4. 控制闭环的计算图

现在把这个机制应用到我们的控制系统。一次仿真（比如 3 步）的计算图：

```
                  θ（控制器参数，所有步共享）
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
 ┌──────┐     ┌──────┐     ┌──────┐
 │控制器│     │控制器│     │控制器│
 │step 0│     │step 1│     │step 2│
 └──┬───┘     └──┬───┘     └──┬───┘
    │δ₀          │δ₁          │δ₂         (转向角输出)
    ▼             ▼             ▼
 ┌──────┐     ┌──────┐     ┌──────┐
 │车辆  │────→│车辆  │────→│车辆  │
 │step 0│state│step 1│state│step 2│
 └──┬───┘  ₁  └──┬───┘  ₂  └──┬───┘
    │             │             │
    ▼             ▼             ▼
  error₀       error₁       error₂

    └──────────┬──────────────┘
               ▼
      loss = Σ errorᵢ²
```

注意两个关键点：

1. **θ 被所有步共享**——同一组参数在每一步都被使用（类似 RNN 的权重共享）
2. **state 向右传播**——step 0 的输出（state₁）是 step 1 的输入

---

## 5. 一步之内：控制器如何使用参数

以横向控制器的一个查找表为例。假设 T2（prev_time_dist 表）有如下断点：

```
速度 (X, 固定):  [0, 10, 20, 30, 40, 50, 60]  km/h
输出 (Y, 可调):  [y₀, y₁, y₂, y₃, y₄, y₅, y₆]   ← 这些是 nn.Parameter
```

当车速为 25 km/h 时，落在 x=20 和 x=30 之间，`lookup1d` 做**线性插值**：

```
t = (25 - 20) / (30 - 20) = 0.5

prev_time_dist = y₂ × (1 - 0.5) + y₃ × 0.5
               = 0.5 × y₂ + 0.5 × y₃
```

这是一个简单的加权求和。导数显而易见：

```
∂(prev_time_dist) / ∂y₂ = 0.5
∂(prev_time_dist) / ∂y₃ = 0.5
∂(prev_time_dist) / ∂y₀ = 0       （不在当前插值区间内）
```

然后 `prev_time_dist` 继续参与后续计算（影响预瞄距离 → 影响 target_theta → 影响转向角），每一步都是可导的数学运算（atan、sin、cos、乘除），PyTorch 自动记录并在 backward 时链式求导。

---

## 6. 跨步传播：间接影响如何产生

这是最关键的部分。θ 对 loss 的贡献分两种：

### 6.1 直接影响

```
θ ──→ 控制器(step t) ──→ δₜ ──→ errorₜ ──→ loss
```

step t 时，θ 直接决定了转向角 δₜ，δₜ 和参考轨迹的偏差构成 errorₜ。

这条路径的梯度很短，只经过当前步内的计算。

### 6.2 间接影响

```
θ ──→ 控制器(step 0) ──→ δ₀ ──→ state₁ ──→ state₂ ──→ ... ──→ stateₜ ──→ errorₜ ──→ loss
```

step 0 时，θ 决定了 δ₀，δ₀ 改变了车辆状态 state₁，state₁ 又影响了 step 1 的控制器输入，
从而影响 δ₁，进而影响 state₂ ... 一直传播到 step t。

**这就是"蝴蝶效应"在梯度中的体现**：早期的参数选择通过状态传播影响了所有后续步骤。

### 6.3 用自行车模型的公式看传播

车辆 step 的核心方程：

```
yaw_{t+1} = yaw_t + v_t × tan(δ_t) / L × dt
```

对 δₜ 求偏导：

```
∂yaw_{t+1} / ∂δ_t = v_t / (L × cos²(δ_t)) × dt
```

假设 v = 5 m/s，L = 3.5 m，δ = 0.05 rad，dt = 0.02 s：

```
∂yaw_{t+1} / ∂δ_t = 5 / (3.5 × cos²(0.05)) × 0.02 ≈ 0.0286 rad/rad
```

这意味着：δₜ 变化 1 rad，下一步的 yaw 变化约 0.0286 rad。

**跨多步的间接影响 = 每步的传播系数连乘**：

```
∂stateₙ / ∂state₀ = ∏(i=0 到 n-1) ∂state_{i+1} / ∂stateᵢ
```

这是一个矩阵连乘（因为 state 有 4 个分量 x, y, yaw, v）。

---

## 7. PyTorch 如何自动完成这一切

你**不需要手动推导**上面的公式。PyTorch 的工作方式：

### Forward（前向）

```python
for step in range(N):
    steer = controller.compute(car.x, car.y, ...)    # 每步运算自动记录
    car.step(delta=steer/ratio, acc=acc_cmd)          # 更新状态，自动记录
    history.append({'lateral_error': ..., ...})
```

每一行 tensor 运算都在计算图中添加一个节点。N 步后，内存中有一个巨大的 DAG（有向无环图）。

### Backward（反向）

```python
loss = tracking_loss(history)    # 汇总所有步的误差
loss.backward()                  # 一次调用，自动遍历整个图
```

`backward()` 从 loss 出发，按**逆拓扑序**遍历每个节点：

1. 到达 `error₂` → 知道 `d(loss)/d(error₂) = 1`
2. 到达 `state₃`（从 error₂ 来）→ 用链式法则算 `d(loss)/d(state₃)`
3. 到达 `δ₂`（从 state₃ 来）→ 用 `∂state₃/∂δ₂` 算 `d(loss)/d(δ₂)`
4. 到达 `θ`（从 δ₂ 来）→ **累加** `d(loss)/d(θ) += d(loss)/d(δ₂) × ∂δ₂/∂θ`
5. 同时到达 `state₂`（从 state₃ 来）→ 继续往前传
6. 在 `state₂` 处，**来自 error₂ 和 error₁ 的梯度汇合、累加**
7. 继续往前 ... 直到遍历完所有节点

最终 `θ.grad` 包含了**所有步、所有路径（直接+间接）的梯度总和**。

### 一句话总结

> **Forward 时 PyTorch 偷偷建图，backward 时自动沿图求导。你只管写 forward 代码。**

---

## 8. 参数更新（梯度下降）

有了梯度之后，用优化器更新参数：

```python
optimizer = Adam(params, lr=0.001)

for epoch in range(n_epochs):
    optimizer.zero_grad()                # 清空上一轮的梯度
    history = run_simulation(...)        # forward：跑仿真，建图
    loss = tracking_loss(history)        # 计算 loss
    loss.backward()                      # backward：反向求导
    optimizer.step()                     # 用梯度更新参数：θ ← θ - lr × dL/dθ
```

每轮迭代：
- `θ.grad` 告诉我们"loss 关于 θ 的斜率"
- 如果 grad > 0，说明 θ 增大会让 loss 增大 → 应该**减小** θ
- 如果 grad < 0，说明 θ 增大会让 loss 减小 → 应该**增大** θ
- 优化器按 `θ_new = θ - lr × grad` 更新

经过多轮迭代，参数逐步收敛到使 loss 最小的值。

---

## 9. 梯度爆炸与 Truncated BPTT

### 9.1 为什么会梯度爆炸

间接影响的梯度 = N 个 Jacobian 矩阵连乘：

```
∂stateₙ / ∂state₀ = J_{n-1} × J_{n-2} × ... × J₁ × J₀
```

如果每个 Jacobian 的"放大倍数"（最大特征值）> 1，连乘 N 次后会**指数增长**。

类比：`1.01³⁰⁰ ≈ 19.8`，`1.05³⁰⁰ ≈ 2,273,996`。

当 N=500（10 秒仿真 × 50Hz），即使每步只放大一点点，梯度也会爆炸到 NaN。

### 9.2 Truncated BPTT 的做法

每隔 K 步，**切断状态的计算图链接**：

```python
if step % K == 0:
    car.detach_state()    # state 不再记得它是怎么算出来的
```

效果：

```
[step 0 ─── step K-1]  ✂  [step K ─── step 2K-1]  ✂  [step 2K ─── step 3K-1]
     段 0                       段 1                       段 2
```

- **直接影响不受影响**：每一步的 θ → δₜ → errorₜ 路径完好
- **间接影响被截断**：梯度最多往前传 K 步就停了

```
没有截断：step 0 的 θ 通过 state 链影响 step 299 的 error（连乘 299 次）
K=64 截断：step 0 的 θ 最多影响 step 63 的 error（连乘 63 次）
```

这是**精度和稳定性的权衡**：
- K 太大 → 梯度爆炸
- K 太小 → 丢失长程依赖信息
- 实践中 K=32~128 通常是好的起点

---

## 10. 本项目中的具体应用

### 哪些参数在优化

| 参数 | 数量 | 梯度来源 |
|------|------|----------|
| 横向 T1-T8 查找表 Y 值 | 65 | 通过 lookup1d 线性插值 |
| 横向 kLh | 1 | 直接乘法 |
| 纵向 L1-L5 查找表 Y 值 | 31 | 通过 lookup1d 线性插值 |
| 纵向 PID 增益 (kp, ki) | 6 | 通过 PID.control() 乘法 |
| 纵向 switch_speed | 1 | 通过 smooth_step 混合 |

### 不可微操作的处理

| 操作 | 问题 | 处理方式 | 梯度行为 |
|------|------|----------|----------|
| `rate_limit`（硬 clamp） | 被 clamp 时梯度=0 | Straight-Through Estimator | forward 硬限幅，backward 无条件传梯度 |
| `if/else` 分支 | 不连续 | `smooth_step`（sigmoid 混合） | 两个分支的加权平均，权重可导 |
| `sign(x)` | x=0 处不可导 | `smooth_sign = tanh(x/temp)` | 处处可导的近似 |
| `max(a, b)` | 拐点不可导 | `smooth_lower_bound`（softplus） | 处处可导的近似 |
| `argmin`（最近点查找） | 离散选择 | `detach` 隔离 | 选择哪个点不传梯度，误差计算传梯度 |

### 训练与验证的分离

```
训练（differentiable=True）：使用上述平滑近似 → 梯度可流通 → 优化参数
                                          ↓
                                    得到优化后的参数 θ*
                                          ↓
验证（differentiable=False）：使用原始硬限幅逻辑 → 确认参数在真实控制器中有效
```

平滑近似只在训练时使用，验证时回到和 controller_spec.md 完全一致的硬限幅逻辑。
