# 纵向控制器扭矩输出层设计

**日期**：2026-04-15
**状态**：设计完成，待实施

## 背景

当前 sim 中纵向控制器 `lon.py` 输出加速度指令 `acc_cmd` (m/s²)，被控对象 `dynamic_vehicle / hybrid_dynamic_vehicle` 用极简公式 `T = (m·a/2)·R` 把 acc 转成扭矩喂给底层。这与真实 C++ 控制器的输出链路不一致——真实 C++ 会按完整物理公式（含风阻、滚阻、惯性力、加速度闭环 P 补偿）算出扭矩。

plant MLP 在外部仓库训练时使用"控制器输出扭矩"作为输入。sim 若想让 plant 外推分布对齐，需要让控制器的扭矩输出规则与真实 C++ 一致。

## 目标

在 `lon.py` 末尾加一个"加速度→车轮扭矩"转换层，公式与 C++ `CalFinalTorque` 逐项对齐（除了跳过传动比/效率那一步，因为我们到车轮扭矩就截止）。控制器原有 Steps 1-6（算 `acc_cmd`）不变。

`dynamic_vehicle / hybrid_dynamic_vehicle` 的 `step()` 接口直接改为吃扭矩，不保留旧的吃加速度的签名（反正旧的 `(m·a/2)·R` 逻辑就是占位桥接，没有独立物理意义）。

新增的 `compute_torque_wheel()` 方法必须遵循 `lon.py` 现有约定提供两条路径：
- `_compute_v1` — 非可微，用于 V1 行为验证和对照真实 C++ 输出
- `_compute_differentiable` — 可微，用于梯度调参

## 非目标

- 不加坡度估计（F_slope = 0）
- 不做扭矩饱和上限（先不限）
- 不做档位状态机
- 不改 kinematic 模型的接口（它天然只能吃加速度，和扭矩层不冲突）
- plant 训练侧如何对齐由 plant 仓库负责，sim 这边不考虑
- 不做"保留旧 acc 链路"的向后兼容开关

## 公式定义

对齐 `lon_controller.cc` L749-811，简化掉坡度和传动比：

```
F_air     = 0.5 × coef_cd × air_density × frontal_area × v²
F_rolling = coef_rolling × veh_mass × 9.81
F_slope   = 0                                                # 不做坡度
F_inertia = coef_delta × veh_mass × acc_cmd
F_resist  = F_air + F_rolling + F_slope + F_inertia

F_P       = accel_to_torque_kp × (acc_cmd − a_actual)

T_wheel_raw = (F_resist + F_P) × wheel_rolling_radius        # 路线 Y：不除 η/i_gear

if acc_cmd > -0.05:     # kAccelerationDeadZone
    T_wheel_out = T_wheel_raw
else:
    T_wheel_out = 0     # 低于死区进入制动模式，扭矩为 0
```

**输入**：
- `acc_cmd` — 控制器上游算出（已有）
- `v` (m/s) — 当前车速（已有）
- `a_actual` (m/s²) — 当前实际加速度，新增输入。sim 里从被控对象状态差分 `(v[t] − v[t−1]) / dt` 得到

**输出**：`T_wheel_out` (N·m) — 车轮总扭矩。后驱两轮时下游分半 `T_rl = T_rr = T_wheel_out / 2`

## 参数表

新增 yaml 段 `lon_torque:`，全部为 buffer（不参与梯度优化）。全部取 C++ 真实值。

| 参数 | 值 | 来源 |
|---|---|---|
| `veh_mass` | 9300 | `lon_controller.cc` L687 默认值 |
| `coef_cd` | 0.6 | `control.conf` L41 |
| `coef_rolling` | 0.013 | `control.conf` L40 |
| `coef_delta` | 1.05 | `control.conf` L42 |
| `air_density` | 1.2041 | `lon_controller.cc` L677 硬编码 |
| `gravity` | 9.81 | `lon_controller.cc` L678 硬编码 |
| `frontal_area` | 9.7 | 用户确认 |
| `wheel_rolling_radius` | 0.5 | 用户确认 |
| `accel_to_torque_kp` | 1000 | `control.conf` L45 |
| `accel_deadzone` | -0.05 | `lon_controller.cc` `kAccelerationDeadZone` |

## 可微参数变化对照

| 范畴 | 原有 | 新增 |
|---|---|---|
| 可微（nn.Parameter）| 横向 T2/T3/T4/T6 + 纵向 7 PID + switch_speed | 无 |
| 固定（buffer）| 横向 T1/T5/T7/T8 + kLh + 速率限；纵向 L1-L5 + IIR α + 预览窗等 | 纵向扭矩层 10 个物理常数 |

**调参维度完全不变**，扭矩层纯粹是物理量换算。

## 实施清单

### 代码改动

| 文件 | 改动 |
|---|---|
| `sim/controller/lon.py` | 构造函数读取 `cfg['lon_torque']` 注册为 buffer；新增方法 `compute_torque_wheel(acc_cmd, speed_mps, a_actual) → T_wheel`，**必须同时实现 `_compute_v1` 和 `_compute_differentiable` 两条路径**（与现有 `compute()` 约定一致） |
| `sim/model/dynamic_vehicle.py` | `step(delta, acc)` → `step(delta, torque_wheel)`；去掉 L168 的 `(m·a/2)·R`；直接 `T_rl = T_rr = torque_wheel / 2`。**不保留旧的 acc 签名** |
| `sim/model/hybrid_dynamic_vehicle.py` | 同上 |
| `sim/sim_loop.py` | 按 `plant_type` 分支：kinematic 传 `acc_cmd`，dynamic/hybrid 调用 `compute_torque_wheel(...)` 后传 `torque_wheel`。同时负责维护 `v_prev` 以计算 `a_actual` |
| `sim/configs/default.yaml` | 新增 `lon_torque:` 段（10 个字段） |

### 不动

- `sim/model/vehicle.py`（kinematic）— 保留 `step(delta, acc)` 接口
- `sim/controller/lat_truck.py` — 横向无变化
- `sim/optim/train.py` / `post_training.py` — 训练脚本无需改动（调参维度未变）

### 可微性处理

整条链路只有一处非连续——死区 `if acc_cmd > -0.05`，其余（F_air/F_rolling/F_inertia/F_P/× 半径）都是纯多项式或线性运算，直接可微。

死区的处理：**不做 smooth 近似，直接用 detach 的指示掩码**：

```python
mask = (acc_cmd > -0.05).float().detach()   # 指示函数脱离梯度图
T_wheel_out = mask * T_wheel_raw
```

理由（参考 `sim/CLAUDE.md` L168 / `docs/bptt_gradient_explosion_analysis.md`）：
- smooth_step 的 temperature 选小了链式相乘会梯度爆炸，选大了又和硬截断偏差大，历史上已踩过坑
- 死区在训练中的梯度贡献 **本来就不应该有**——当 `acc_cmd ≤ -0.05` 时扭矩通道对 plant 无贡献（物理上由刹车系统处理），梯度为 0 是正确的
- 正常训练轨迹（lane_change/clothoid/s_curve）不会密集穿越 -0.05 这个阈值，mask 大部分时候恒为 1 或恒为 0，不影响 BPTT 稳定性

`a_actual` 通过车速差分获取，上一步速度 detach：

```python
v_prev = vehicle.v.detach()      # 上一步速度（detach，不回传）
# ... 控制器 compute 输出 acc_cmd ...
a_actual = (vehicle.v - v_prev) / dt
```

首帧 `a_actual = 0`。梯度分析：当前步的 `acc_cmd[t]` 还没作用到 `v[t]` 上（plant.step 此时尚未调用），所以 `∂a_actual / ∂acc_cmd[t] = 0`，P 项不引入新的梯度反馈回路。

## 验证计划

1. **数值一致性**：离线用相同 `acc_cmd/v/a_actual` 输入，对比 sim `compute_torque_wheel` 与 C++ 公式的输出，误差应在 1e-4 以内
2. **baseline 回归**：`run_demo.py --plant kinematic` 结果应与改动前完全一致（kinematic 路径未改）
3. **扭矩量级诊断**（重点）：在 lon.py 里加日志打印各分项 `F_air / F_rolling / F_inertia / F_P`：
   - kinematic 闭环下 `Error_Ax ≈ 0`（kinematic 的 `v += acc·dt`，差分加速度本应等于上一步 acc），P 项应极小
   - 如果 P 项占总扭矩 > 10%，说明模型不匹配严重
4. **动力学闭环**：`run_demo.py --plant hybrid_dynamic --save`，肉眼确认轨迹跟踪不退化
5. **训练收敛**：`train.py --plant hybrid_dynamic --epochs 6 --trajectories lane_change` 跑一遍，检查 loss 下降幅度和之前 baseline 对比
6. **梯度健康**：检查 T6 等已知梯度信号在新路径下是否仍有非零 grad_norm

## 风险

- **风险 1（主要）**：控制器扭矩用 9300 kg 卡车参数算出，若 plant MLP 的训练数据分布基于不同车重，扭矩量级不匹配导致 plant 外推异常。
  **缓解**：plant 训练侧会对齐控制器输出（用户已确认）。sim 内先用 kinematic 跑通整条链路，再切换到 hybrid_dynamic
- **风险 2**：`a_actual` 通过速度差分引入数值噪声，P 补偿项不稳。
  **缓解**：sim 仿真步长固定 0.02s、速度是 float 精度，数值差分噪声可控。如仍有问题，可改为从被控对象直接读取（加 `vehicle.last_acc` 属性）
- **风险 3**：死区 mask 在阈值附近频繁切换，导致扭矩输出有高频抖动。
  **缓解**：实测中观察，若确实抖动严重再考虑加 hysteresis（但会增加状态）

## 后续工作（本文档不覆盖）

- 若训练目标扩展到扭矩层参数（`coef_cd / coef_rolling / accel_to_torque_kp`），再讨论是否让这些参数变为 nn.Parameter
- 若要加扭矩饱和上限，需要先确认 C++ 端的真实上限（目前仓库里的 1800 Nm 是发动机扭矩上限，车轮扭矩上限需要传动比信息）
- 若要复现档位逻辑（N/D/R/P 切换），作为独立任务
