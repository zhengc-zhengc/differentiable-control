# BPTT 梯度爆炸分析：可微闭环控制中的 smooth 近似温度问题

## 1. 现象

对可微控制器进行 BPTT 训练时，梯度出现 NaN/Inf：

| 参数 | 梯度量级 | 状态 |
|------|---------|------|
| lat_ctrl.T2_y | `inf` | 被 nan_to_num 置零，完全不更新 |
| lat_ctrl.T3_y | `inf` | 同上 |
| lat_ctrl.T4_y | `inf` | 同上 |
| lat_ctrl.T5_y | 无梯度 | `.item()` 断开了计算图 |
| lat_ctrl.T6_y | 无梯度 | 同上 |
| lon_ctrl.station_kp | 1.7e18 | 被 clamp 到 1e4 |
| lon_ctrl.high_speed_ki | 8.9e18 | 同上 |

但 **loss 本身完全正常**（~65，有限且合理），系统是稳定的闭环控制，小参数扰动不会导致 loss 爆炸。

## 2. 关键认知：梯度爆炸 ≠ Loss 爆炸

用有限差分验证了 loss 景观是光滑的：

```
参数: lon_ctrl.high_speed_kp = 0.34
有限差分梯度:  -19.87     ← 真实梯度，完全合理
BPTT 梯度:     NaN        ← 数值溢出
```

**真实梯度只有 -19.87，BPTT 算出 NaN** — 问题不在 loss 景观，而在 BPTT 的数值计算方式。

## 3. 根因：smooth 近似的 temperature 过小

### 3.1 BPTT 的链式乘法

BPTT 计算梯度的方式是逐步相乘局部 Jacobian：

$$\frac{\partial L}{\partial \theta} = \sum_t \frac{\partial L}{\partial s_t} \cdot \prod_{k=t}^{T-1} \frac{\partial s_{k+1}}{\partial s_k} \cdot \frac{\partial s_t}{\partial \theta}$$

每一步的局部 Jacobian $\frac{\partial s_{k+1}}{\partial s_k}$ 如果 > 1，乘 N 步就指数增长。

### 3.2 smooth 近似函数的局部导数放大

为了让硬非线性（sign、clamp、step）可微，我们用 smooth 近似替代。但这些函数在过渡区有很大的局部导数：

| 函数 | 公式 | temp | 过渡区最大导数 |
|------|------|------|--------------|
| `smooth_sign(x, temp)` | `tanh(x/temp)` | 0.01 | **100** |
| `smooth_clamp(x, lo, hi, temp)` | `mid + half·tanh(...)` | 0.1 | **~10** |
| `smooth_step(x, thr, temp)` | `sigmoid((x-thr)/temp)` | 0.001 | **250** |

### 3.3 链式乘法的指数爆炸

在一个时间步内，信号依次经过：
1. `smooth_sign` (横向 Step 6) → 100x 放大
2. `smooth_clamp` (横向 Step 10) → 10x 放大
3. 单步合计：~1000x

BPTT 链式乘法跨 N 步：

| 每步放大 | 32 步乘积 | 结果 |
|---------|----------|------|
| 100x | 100^32 | 10^64 → **Inf** |
| 10x | 10^32 | 10^32 → **Inf** |
| 2x | 2^32 | 4.3e9 → 勉强可用 |
| 1.5x | 1.5^32 | 4.3e5 → 可用 |

### 3.4 为什么系统是稳定的但梯度爆炸？

闭环控制系统的 forward pass 中，smooth 函数的过渡区通常只在信号过零的瞬间被激活（比如横向误差为零时 smooth_sign 的导数 = 100），大部分时间信号在饱和区（导数 ≈ 0）。所以 **forward 轨迹是有界的**。

但 backward pass 的链式乘法会捕获那些少数过零时刻的巨大局部导数，并将其与其他步的导数相乘。即使只有几步导数很大，在乘积中也会主导整体梯度。

**类比**：想象一根绳子，大部分地方直径 1cm，但有 3 处膨胀到 100cm。绳子整体能穿过大部分隧道（forward pass 稳定），但总体积（backward pass 的 Jacobian 乘积）被那 3 处主导。

## 4. 具体问题位置

### 4.1 横向控制器 `lat_truck.py`

```python
# Step 6: target_theta — smooth_sign temp=0.01 → 导数 100x
target_theta = smooth_sign(error_angle_raw, temp=0.01) * max_err_angle

# Step 4: vehicle_speed_clamped — smooth_clamp temp=0.1 → 导数 ~10x
vehicle_speed_clamped = smooth_clamp(speed_mps, 1.0, 100.0, temp=0.1)

# Step 10: steer 输出范围限制 — smooth_clamp temp=0.1 → 导数 ~10x
steer_raw = smooth_clamp(steer_fb + steer_ff, -max_steer_angle, max_steer_angle, temp=0.1)
```

### 4.2 纵向控制器 `lon.py`

```python
# Step 6: 曲率条件 — smooth_step temp=0.001 → 导数 250x !!
w_curv = smooth_step(-curvature_far, 0.0075, temp=0.001)
```

### 4.3 T5/T6 无梯度问题（独立问题）

```python
# lat_truck.py Step 2: .item() 断开了 T5/T6 的梯度链
near_pt_time_val = near_pt_time.item()   # ← 梯度在此截断
far_pt_time_val = far_pt_time.item()     # ← 梯度在此截断
```

T5/T6 的值用于查询轨迹点的时间偏移。因为 `TrajectoryAnalyzer` 的查询接受 float 而非 tensor，必须 `.item()` 转换，这导致梯度无法回传到 T5/T6 的 nn.Parameter。

## 5. 解决方案

### 5.1 提升 smooth 近似的 temperature

将过渡区导数控制在 ~2x 以内（32 步链乘 ≈ 2^32 ≈ 4e9，配合 gradient clipping 可用）：

| 函数 | 修改前 temp | 修改后 temp | 修改前导数 | 修改后导数 |
|------|-----------|-----------|----------|----------|
| `smooth_sign` (lat Step 6) | 0.01 | **0.5** | 100 | **2** |
| `smooth_clamp` (lat Step 4) | 0.1 | **1.0** | ~10 | **~1** |
| `smooth_clamp` (lat Step 10) | 0.1 | **1.0** | ~10 | **~1** |
| `smooth_step` (lon Step 6 曲率) | 0.001 | **0.01** | 250 | **25** |

### 5.2 添加全局梯度裁剪

在 `train.py` 中 `loss.backward()` 之后、`optimizer.step()` 之前：

```python
grad_norm = torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=10.0)
```

### 5.3 修复效果

| 参数 | 修复前梯度 | 修复后梯度 |
|------|----------|----------|
| lat_ctrl.T2_y | `inf` | 0.088 |
| lat_ctrl.T3_y | `inf` | 0.711 |
| lat_ctrl.T4_y | `inf` | 0.645 |
| lon_ctrl.station_kp | 1.7e18 | 0.235 |
| lon_ctrl.high_speed_kp | 1.4e18 | 0.070 |
| lon_ctrl.high_speed_ki | 8.9e18 | 4.097 |

**BPTT 梯度 vs 有限差分验证**（`high_speed_kp`）：
- 有限差分：0.134
- BPTT：0.070
- 比值：0.52（同量级一致，smooth 近似引入的偏差可接受）

## 6. 进一步优化：消除不必要的 smooth 近似

### 6.1 发现 `sign(x) * min(|x|, L) ≡ clamp(x, -L, L)`

横向控制器 Step 6 的 C++ 原始代码用 `sign * min(abs, limit)` 分解实现限幅：

```python
# C++ 风格分解（不必要的复杂）
abs_error = abs(error_angle_raw)
max_err_angle = min(abs_error, max_theta_limit)
target_theta = sign(error_angle_raw) * max_err_angle
```

可微复现时，逐个替换为 smooth 版本（smooth_sign + smooth_min + abs），引入了 smooth_sign 的高导数问题。

但这整个操作数学上等价于：

```python
# 等价的简单形式
target_theta = clamp(error_angle_raw, -max_theta_limit, max_theta_limit)
```

用 `_straight_through_clamp`（STE clamp）替代后：
- 梯度在 clamp 范围内恒为 **1**（无放大）
- BPTT 与有限差分比值：T2 → **0.91**，T3 → **1.10**（接近完美）
- 消除了 smooth_sign 这个最大的梯度爆炸源

### 6.2 启示

在将硬限幅代码改为可微版本时，应先**化简数学表达式**，再选择 smooth 近似。不必忠实翻译 C++ 的每一步分解——先看整体做了什么运算，用最简单的可微形式实现。

## 7. 设计原则总结

1. **先化简再近似**。将硬限幅代码改为可微版本前，先检查整体运算是否有更简单的等价形式（如 `sign*min(|x|,L)` → `clamp`）。

2. **smooth 近似的 temperature 必须足够大**，使局部导数 ≤ 2-3x。`temp = 0.01` 这种数值几乎必然导致梯度爆炸。

3. **有限差分是 BPTT 梯度的 ground truth**。当怀疑梯度异常时，用 `(f(x+ε)-f(x-ε))/(2ε)` 验证。

4. **梯度爆炸 ≠ loss 爆炸**。闭环控制系统的 loss 景观通常是光滑有界的，但 BPTT 的链式乘法可能数值溢出。

5. **gradient clipping 是安全网**，但不应该是主要手段。如果需要 clip 才能训练，说明 smooth 近似有问题。

6. **`tbptt_k`（截断窗口）控制梯度链长度**，越短越稳定但越短视。修复梯度问题后通常可以用较长的窗口（64-128 步）。
