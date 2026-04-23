# 轨迹跟踪持续稳态误差 —— 原因诊断

**分支**：`debug/tracking-steady-state`  
**日期**：2026-04-23  
**触发**：truck_trailer plant 下的轨迹跟踪图出现持续的稳态横向偏差，在 R=30 m 圆弧上 20 s 后能扩大到 ±60 m 级别，远不是"稳态残差"那么简单。问题："反馈+前馈的控制器应该很鲁棒，为什么跟不上？"

## TL;DR — 一句话回答

这个控制器的"反馈"是 **P + D（没有 I）**，"前馈"是 **纯 Ackermann 运动学公式（零速近似）**；默认配置里还有一个 `T1_max_theta_deg = 3.86°` 的硬限幅，把 P 项的权限在横向误差超过约 0.5 m（v=5 m/s 时）之后整个封顶。被控对象从 2440 kg 乘用车换成 9300 kg 卡车以后，前馈先差一截，反馈的 P 项一碰到 T1 就饱和，剩下的路基本靠前馈"开环"走——所以横向误差只会增长到 plant 的侧偏/横摆自平衡点，无法被抹平。"反馈+前馈应该鲁棒"的前提是反馈里有 **I** 项；这里没有。

## 复现证据

脚本：`debug_trace.py`（同目录）。plant=truck_trailer、圆弧 R=30 m、v=5 m/s，20 s 弧长。

同一条圆弧，换 plant：

| Plant              | 25 s 时 \|lat_err\|  | 末端 steer_fb | 注释                         |
| ------------------ | -------------------- | ------------- | ---------------------------- |
| kinematic          | **0.015 m**          | 0（不用反馈） | 前馈 ≡ 被控对象，P+D 冗余    |
| dynamic（乘用车）  | **0.06 m**           | ≈0            | 前馈小偏差，P+D 足够抹平     |
| **truck_trailer**  | **≈11 m，持续扩大**  | 50–75（饱和） | 误差把 T1 顶住，反馈残留不动 |

## 为什么"P+D+Ackermann 前馈"吃不下 plant 换型

### 1. 横向反馈没有 I 项

`controller/lat_truck.py` Step 6-7：

- Step 6：`target_theta = atan(-lat_err / prev_dist)`（**P on lateral**，再被 Step 6 里的 T1 clamp 卡住）
- Step 7：`target_curvature = ((target_θ − real_θ) + (target_dt_θ − real_dt_θ)·T_dt) / (T3·v)`（P + D on 航向/航向率误差）
- Step 9：`steer_ff = atan(κ_far · wheelbase) · RAD2DEG · steer_ratio · slip_param`（纯 Ackermann）

**没有 `∫ lat_err dt`**。如果前馈和 plant 之间存在一个常值偏差，反馈靠 P 补偿时必然留一个非零残差，残差大小 = 前馈偏差 ÷ 等效 P 增益。

### 2. T1 = 3.86° 把 P 项硬封顶

```
target_theta = clamp(atan(-lat_err / prev_dist), ±3.86°)  =  clamp(..., ±0.0674 rad)
```

在 v = 5 m/s、T2 = 1.5 → `prev_dist = 7.5 m`。只要 `|lat_err| > 7.5·tan(3.86°) ≈ 0.505 m`，`target_theta` 就顶到 ±0.0674 rad 不再增长。

此后 P 项提供的 `target_curvature` 再怎么扩大 lat_err 都不涨了，反馈只剩 D 项对航向率误差的一点点修正——实际上"闭环回路失能"，变成前馈单干。

**实验 B（`debug_hypotheses.py`）**：把 T1 从 3.86° 抬到 45°（其他不动），同样的 truck_trailer 圆弧，最大 \|lat_err\| 从 **13.05 m → 1.61 m**（8 倍改善）。这是最强力的单变量证据。

### 3. Ackermann 前馈对卡车"结构性偏离"

FF = `atan(κ · L)` 是 **零速极限下** 的稳态 Ackermann 关系，假定无轮胎侧偏、无横摆惯量、无铰接耦合。卡车动力学里全部都有：

- Iz_t = 48639 kg·m²（≈ 乘用车的 5 倍），前馈指令阶跃进入 plant 时横摆响应有明显时间常数
- 前后轴 lf=3.8 vs lr=0.675 极度不对称（CoM 贴后轴 — 铰接构型典型），前后轴轮胎负载差异 × Cα 有限 → 稳态有非零侧偏
- 即使 `default_trailer_mass_kg = 0`（当前默认），底层 12D 状态积分里挂车块仍参与（之后才被适配器 clamp 回牵引车态），数值上可能留下小扰动

实测：truck_trailer 圆弧稳态段 FF = 139 deg，但 plant 实际维持圆弧所需的总 steer ≈ 195 deg（实验 A 末端 `steer_total`），**FF 差了约 56 deg 的方向盘角 / 3.4° 的前轮角**。线性 2-DOF 自行车模型给出的不足转向梯度 K_us 在这个 lf/lr 构型下实际上是过转 / K_us<0 的符号（理论上应该是 FF *比需要的多*），与观测方向相反 — 说明主导项不是线性轮胎侧偏，而是更高阶效应（Iz 横摆瞬态、非线性 slip、或 trailer 状态积分残留）。结论是不论具体由谁主导，**FF 与 plant 之间存在约 3° 前轮角的系统性偏差**，需要反馈补偿，而 T1 限死之后反馈最多只能补约 75 deg → 仍差约 20 deg，对应 \|lat_err\| 在 10 m 量级飘。

### 4. sim_loop 喂给控制器的 yawrate 是"合成值"，不是 plant 真值

`sim_loop.py:101` / `:176`：

```python
delta_prev = prev_steer / steer_ratio * DEG2RAD
yawrate = car.v * math.tan(delta_prev) / wheelbase   # 用 kinematic bicycle 反推
```

对 kinematic plant 这就是 plant 的真 yawrate；对 dynamic / truck_trailer plant，plant 的真 yawrate（state[5]=`r_t`）来自动力学积分，受 Iz、Cα、轮胎 slip 影响，**普遍比 kinematic 的公式值小**。

`debug_yawrate.py` 实测对比（truck_trailer 圆弧稳态段）：

| t (s) | yawrate_synth (rad/s) | yawrate_plant (rad/s) | κ_ref·v (期望) |
| ----- | --------------------- | --------------------- | -------------- |
| 1     | 0.31                  | **0.16**              | 0.17           |
| 5     | 0.24                  | **0.15**              | 0.16           |
| 10    | 0.26                  | **0.15**              | 0.17           |

Step 5 `real_dt_theta = yawrate − κ_far·v`：

- 用 synth：`real_dt_theta ≈ +0.09 rad/s` → Step 7 D 项解读为"我转得比曲率要求还快，减小打角"
- 用 plant 真值：`real_dt_theta ≈ −0.02 rad/s` → 解读为"我转得略慢，加一点打角"

**D 项的方向都反了。** 不过单独改这一项效果比 T1 小（见表），因为 P 项已经被 T1 顶住，D 的贡献本来就是二阶的。

### 5. 纵向链路：station 环 I 被关，速度环 I 权限也很小

`configs/default.yaml`：

```yaml
lon:
  station_kp: 0.25
  station_ki: 0.0
  station_integrator_enable: false      # 站位误差完全没有 I
  ...
  low_speed_ki: 0.01
  high_speed_ki: 0.01
  speed_pid_sat: 0.3                    # 速度环 I 项累积上限 ±0.3 m/s
```

实验 A 末端速度 = 6.07 m/s（参考 5.0 m/s）。机理：横向发散→车走 s 方向"绕远路"→station_error 持续累积（但 ki=0 所以仅 P 导出）→ station_kp × 误差 → 给速度环一个正的 speed_offset → 速度 PID 的 I 权限（0.3 m/s）拉不回这个偏置 → 稳态速度偏高。

## 四个成因按影响排序

用 truck_trailer R=30 v=5 圆弧做最小变量对照（单独改一项，其他不动）：

| 实验 | 改动                                  | \|lat_err\|max  | 改善幅度    |
| ---- | ------------------------------------- | --------------- | ----------- |
| A    | baseline（现状）                      | **13.05 m**     | —           |
| B    | T1 clamp 3.86° → 45°                  | 1.61 m          | **−88 %**   |
| C    | T3 reach_time 1.1 → 0.4（放 P 增益）  | 8.34 m          | −36 %       |
| D    | B + C                                 | 1.35 m          | −90 %       |
| G2   | 用 plant 真实 yawrate（T1 保持 3.86） | 10.02 m         | −23 %       |
| H2   | uncap T1 + plant 真 yawrate           | **1.51 m**      | −88 %       |

影响排序：**T1 saturation（主因）≫ FF 模型偏差（≈1.3 m 残留） > 合成 yawrate 方向错 ≈ 反馈 P 增益不够**。

## 为什么用户的直觉"反馈+前馈应该鲁棒"在这里不成立

教科书里"PI + FF" 对被控对象变化鲁棒的前提是：

1. **反馈里有 I**，能把任何常值扰动 / 建模偏差积分消除
2. P 项权限足够大，瞬态期能拉回偏差再交给 I
3. 传感器信号是 plant 真值，不是模型预测

这个 C++ 控制器的设计初衷是让量产车**在正常操作域**里舒适可预测：T1 = 3.86° 是"正常车道内横向误差不会超过 0.5 m"的安全假设下给的上界；不加 I 是为了避免长时间误差积分导致转向突变；前馈用 Ackermann 对乘用车足够（K_us 很小，2440 kg、Cαf=Cαr=80000、lf≈lr → K_us ≈ 0）。换到 truck_trailer（9300 kg、lf=3.8 vs lr=0.675、L 增加 28 %），三个前提同时崩掉：

- 前馈偏差大 → P 必须承担更多
- P 被 T1 封死 → 承担不了
- 没 I → 没人兜底
- D 的 yawrate 还是 kinematic 预测，方向反

结果就是 "被控对象的自平衡点 ≠ 参考轨迹"，误差停在自平衡点附近。这不是"反馈+前馈"失效，而是这个控制器的反馈结构**本来就没设计成对 plant 大幅偏离的情形鲁棒**。

## 与 `sim/CLAUDE.md` 的吻合

该文件 L206 已经预警：

> **跟踪性能预期**：当前控制器参数针对 2440 kg 乘用车调校，直接用在卡车（无挂车或带 15 吨挂车）上跟不动，需要专门调参

但只说了"需要调参"，没拆机理。本报告给出的是机理：**不是调参问题，是结构问题**。只改 `configs/tuned/*.yaml` 里的 T2/T3/T4/T6 时间参数和几个 PID 增益（当前的可微参数集合）——动不了 T1 clamp（固定 buffer）、动不了"有没有 I 项"（结构性）、动不了前馈公式。

## 可能的下一步（仅列方向，不实施）

按结构性 → 参数性排序：

1. **lat 控制加 I 项**：在 Step 6 target_theta 之前加一段 `integ_lat += ki_lat · lat_err · dt`（带反 windup），让 DC 增益 → ∞，消除稳态残差。最小侵入。
2. **把 T1_max_theta_deg 升为可调（或者与速度相关）**：当前卡在 3.86°，放到 15–20° 或做成查找表由 post_training 优化。
3. **FF 里加不足转向补偿**：把 `atan(κ·L)` 改成 `atan(κ·(L + K_us·v²))`，K_us 由 dynamic_vehicle 的 lf/lr/Cα/m 解析求出。
4. **把 plant 真值 yawrate 回传给控制器**：在 `sim_loop.py` 里，如果 plant 暴露 `.r` 或 `._state[5]`，优先使用；fallback 到 kinematic 公式。只影响 D 项的符号正确性，定量贡献小但语义上必要。
5. **纵向 station 环打开 integrator_enable、ki 调成 ~0.05**：消除速度稳态偏置。

以上都是讨论用途，实际动手之前需要另起一个设计讨论。

## 附文件

- `debug_trace.py` — 基础复现：truck_trailer vs kinematic 的状态轨迹
- `debug_hypotheses.py` — 实验 A-F：T1 / T3 / plant 对比
- `debug_yawrate.py` — synth vs plant 真值 yawrate 对照
- `debug_real_yawrate.py` — 实验 G/H：把真实 yawrate 回传控制器看效果
- `debug_steer_ratio.py` — 逐层核对传动比一致性（结论：全部一致）
- `validate_current.py` — 把改动固化为默认行为之后的 49 场景验证脚本

## 已落地的改动

> 诊断结论的第 4 条（合成 yawrate 方向错）已作为修复实装到所有路径。

所有 plant 适配器（`BicycleModel` / `DynamicVehicle` / `HybridDynamicVehicle` /
`GenericHybridVehicle` / `TruckTrailerVehicle` + batched 版本）都暴露
`yawrate` property：
- 动力学 plant：直接返回 `state[5]`（r / r_t）
- kinematic plant：`step()` 末尾记录 `v · tan(δ) / L`，对外 property 返回（与原 synth 数值等价）

`sim/sim_loop.py` 两处 + `sim/optim/train_batch.py` 主循环都从 `car.yawrate` 读取，
不再用 `v · tan(δ_prev) / L` 合成。**训练、post_training 的 49 场景 V1 验证、
validate_batch CLI、scalar run_demo 全走同一套**。

49 场景 truck_trailer V1 验证效果（对比改动前的 synth yawrate）：
- 改善 41 条 / 持平 8 条（全部 5 kph，理论自洽）/ 恶化 0 条
- 平均 lat RMSE：**−14.88 %**；平均 head RMSE：**−13.77 %**
- 最大单场景改善 −24.0 %（单换道 55 kph：1.39 m → 1.06 m）

剩下的三条结构性原因（lat 反馈没 I、T1=3.86° 硬限、前馈是 Ackermann 近似）
依然存在，未在本次改动范围内。
