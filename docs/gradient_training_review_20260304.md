# 可微训练流程审查报告（2026-03-04）

## 1. 审查范围

- 基准提交：`1396cdf`（`[sim] 用 STE clamp 替换 smooth_sign：消除横向梯度爆炸根源`）
- 重点文件：
  - `sim/controller/lat_truck.py`
  - `sim/controller/lon.py`
  - `sim/sim_loop.py`
  - `sim/optim/train.py`
  - `sim/common.py`
- 审查目标：
  1) 确认当前训练链路是否仍存在潜在梯度爆炸/消失来源  
  2) 梳理 `item()` 导致的梯度截断  
  3) 梳理并解释“保护操作”具体在做什么，以及是否存在冗余

> 说明：本报告基于代码路径审查和提交记录，不包含本机实时复跑数值（当前审查环境缺少 `torch` 运行时）。

---

## 2. 结论摘要

1. **主爆炸源已修复**：`lat_truck` 的 `smooth_sign` 已被 `STE clamp` 替换，横向主链路导数放大风险大幅下降。  
2. **仍有梯度截断点**：`item()` 在时间查询/Frenet转换/分支判断中仍导致部分路径不可微，主要影响是“学不到”或“梯度过弱”，而非直接爆炸。  
3. **保护层偏多**：当前同时启用了 `TBPTT`、`hook 级 nan/inf 清理 + 元素钳制`、`全局梯度范数裁剪` 三层保护，稳定性高，但可能对有效梯度有过度抑制。

---

## 3. `item()` 导致的梯度截断（核心问题）

## 3.1 横向控制器：T5/T6 路径截断

文件：`sim/controller/lat_truck.py`

```python
near_pt_time_val = near_pt_time.item()
far_pt_time_val = far_pt_time.item()
near = analyzer.query_nearest_by_relative_time(currt.t + near_pt_time_val)
far = analyzer.query_nearest_by_relative_time(currt.t + far_pt_time_val)
```

- `near_pt_time`、`far_pt_time` 来自可训练参数表 `T5_y/T6_y`。
- 一旦 `.item()`，张量变成 Python float，后续不在 autograd 图内。
- 结果：`T5_y/T6_y` 虽是 `nn.Parameter`，但这条时间查询链路中**无法通过反向传播更新**。

**补充说明：`.item()` 只是表层问题，真正的障碍在原理层面。**

即使去掉 `.item()`，`query_nearest_by_relative_time` 内部仍然不可微：

1. 该函数在离散轨迹点列表中做**索引查找**（找到相邻的两个点），这是离散操作，不可微。
2. 插值过程使用 `TrajectoryPoint`（纯 float dataclass）做 float 运算，不在 autograd 图内。
3. 返回的 `TrajectoryPoint` 所有字段都是 Python float，不是 `torch.Tensor`。

要让 T5/T6 真正可微，需要将 `TrajectoryAnalyzer` 的时间查询改为**连续可微插值**——将轨迹的 x(t)、y(t)、theta(t)、kappa(t) 用 tensor 存储，用类似 `lookup1d` 的可微插值返回 tensor 结果。这是架构级改动，不是删一行 `.item()` 能解决的。

对比：T2/T3/T4 的查找表输出直接参与 tensor 运算（不经过轨迹查询），梯度正常流通，不存在此问题。

---

## 3.2 纵向控制器：Frenet 与分支判断中的截断

文件：`sim/controller/lon.py`

```python
t_now_val = t_now.item()
s_match, s_dot, d_frenet, d_dot = analyzer.to_frenet(
    x.item(), y.item(), yaw_rad.item(), speed_mps.item(), matched)
...
elif station_limited.item() <= 0.25:
...
elif station_limited.item() >= 0.8:
```

- `x/y/yaw/speed` 通过 `.item()` 后传给 `to_frenet`，会切断从车辆状态到该分支的梯度链。
- `station_limited.item()` 用于 Python `if`，将连续张量判定离散化，分支切换点不可微，且分支内仅部分路径保留梯度。
- 结果：纵向参数（如 `station_kp`、`switch_speed` 等）并非完全无梯度，但梯度信息会比“全张量可微实现”更稀疏。

---

## 3.3 与 `item()` 无关但会缩短梯度链的相关点（补充）

- `sim/model/trajectory.py` 中最近点查询使用 `detach + no_grad + argmin`，属于设计上接受的“离散索引不可微”隔离。  
- 控制器状态缓存（如 `steer_fb_prev`、`acc_out_prev`）多处用 `detach()` 更新，等价于对状态记忆做截断，减少时序梯度长度。

这类操作不属于 `item()`，但会叠加“梯度链变短”的效果。

---

## 4. 现有“保护操作”具体做了什么

## 4.1 保护层 A：TBPTT（按窗口截断时间链）

文件：`sim/sim_loop.py`

```python
if tbptt_k > 0 and step > 0 and step % tbptt_k == 0:
    car.detach_state()
    prev_steer = prev_steer.detach()
```

作用：
- 每 `K` 步把车辆状态从计算图中分离，反向传播只跨越最近 `K` 步。  
- 显著降低长时链式乘法导致的梯度爆炸/显存增长风险。  

副作用：
- 远期信用分配能力下降（更“短视”），长期效应学习变弱。

---

## 4.2 保护层 B：梯度 Hook（逐参数清理 + 元素级钳制）

文件：`sim/optim/train.py`

```python
_grad_clip_val = 1e4
def _sanitize_grad(grad):
    g = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return g.clamp(-_grad_clip_val, _grad_clip_val)
```

作用：
- 在每个参数的梯度回传到优化器前，先将 `NaN/Inf` 清零，再把每个元素限制到 `[-1e4, 1e4]`。  
- 防止异常梯度污染 Adam 一阶/二阶矩。  

副作用：
- 发生异常时会直接“硬改梯度”，学习信号可能失真；如果常态触发，会掩盖真正的数值问题。

---

## 4.3 保护层 C：全局梯度范数裁剪（按向量范数缩放）

文件：`sim/optim/train.py`

```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    params.parameters(), max_norm=grad_clip).item()
```

作用：
- 如果总体梯度范数超过阈值（当前默认 `10.0`），按比例缩小所有参数梯度。  
- 防止个别 batch/轨迹导致一步更新过大。  

副作用：
- 当长期处于裁剪状态时，会持续压低步长，训练变慢，且可能掩盖上游梯度放大根因。

---

## 5. 为什么说当前存在“冗余保护”倾向

当前链路至少有以下叠加：

1. **时间维度截断**：`TBPTT`（每 64 步）  
2. **参数梯度清理**：`nan_to_num + 元素级 clamp`  
3. **全局更新约束**：`clip_grad_norm_`  
4. **局部状态 detach**：控制器内部状态缓存更新时 detach

在 `smooth_sign -> STE clamp` 修复后，若梯度已经稳定，上述 2+3 往往功能重叠：  
- B 层已经限制了单元素异常，  
- C 层又进一步限制全局范数。  

这不一定“错误”，但可能是“安全网过密”，导致有效梯度被二次压缩。

---

## 6. 建议的最小化验证顺序（消融思路）

建议按以下顺序逐步减保护（每步观察 `grad_norm`、NaN 计数、loss 曲线）：

1. 保持 `TBPTT` 不变，先把 `grad_clip` 调大（如 `100`）或临时关闭，观察是否仍稳定。  
2. 若稳定，再保留 `clip_grad_norm_`，仅放宽 hook 的元素钳制阈值（如 `1e5`）验证触发频率。  
3. 若 hook 几乎不触发，可考虑去掉元素钳制，仅保留 `nan_to_num` 兜底。  
4. 最后再评估是否需要调整 `tbptt_k`（64 -> 96/128）以恢复部分长时信用分配能力。

---

## 7. 本轮优先整改建议

1. **优先级 P0**：处理 `lat_truck` 中 `T5/T6` 的 `item()` 截断（否则这两个参数名义可训练、实际难训练）。  
2. **优先级 P1**：梳理 `lon` 中 `to_frenet` 与 `station` 低速分支的 `item()` 使用，明确哪些必须离散、哪些可替换为连续近似。  
3. **优先级 P1**：对保护层做一次消融实验，确认是否存在“冗余保护导致学习信号过度压缩”。

---

## 8. 一句话总结

当前系统已从“梯度爆炸主导”进入“可训练但有截断和过保护”的阶段：  
**先修 `item()` 造成的关键断链，再做保护层减法，通常能比继续加保护更有效。**

