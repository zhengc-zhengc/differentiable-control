# 激进域随机化：状态反馈噪声 + 指令抖动

**日期**：2026-05-08
**状态**：设计完成，待实施
**分支**：`worktree-experiment+aggressive-dr`
**前置文档**：[2026-05-08-domain-randomization-design.md](2026-05-08-domain-randomization-design.md)
**范围**：在保守档物理参数 DR（`m_t±10% / Cf,Cr±20% / K=4`）之上叠加传感器噪声 + 执行器抖动；仅 `truck_trailer` plant + `train_batch.py`

## 背景与目标

保守档 DR 把不确定性局限在车辆物理常数上，但实车上控制器还要应对两件事：
1. 控制器读到的状态不是真值——`lat_err / heading_err / v_x / yaw_rate` 都带传感器噪声
2. 控制器算出的指令不是被精确执行——ECU 量化、电磁干扰、机构间隙在 `delta / torque` 上叠加高频抖动

不模拟这两件事的训练相当于让控制器 overfit 到"完美感知 + 完美执行"。本设计要把这两层噪声同时打开，与既有物理参数 DR **叠加**起来训练，得到对感知/执行不确定性也鲁棒的控制器参数。

## 范围

**在内**：
- 状态反馈高斯白噪声（4 通道：lat 投影、head、车速、横摆率）
- 指令高频抖动（2 通道：delta、torque_wheel）
- 与现有 `m_t/Cf/Cr` DR 共存（K=4 不变）
- 仅 `train_batch.py` + `truck_trailer` plant
- MLP 开关与噪声开关正交，各自独立由 yaml/CLI 决定

**不在内**：
- 不加传感器零偏 / OU 漂移（仅单步白噪声，均值=0）
- 不加执行器延迟 / rate-limit 漂移（仅 memoryless 抖动）
- 不改 minimax / CVaR 目标（仍是 average loss）
- 不动 scalar `train.py` 路径
- 不动其它 plant（kinematic / dynamic / hybrid_*）

## 噪声与抖动定义

### 状态反馈噪声（中档）

实车上控制器拿到的不是真实位姿，而是定位/IMU/轮速传感器输出。在仿真里把这件事建模为：在控制器读取 `vehicle` 状态之前，往真值上加独立高斯。

| 通道 | 注入对象 | σ | 物理含义 |
|------|---------|---|---------|
| 位置 x | `vehicle.x` | 0.02 m | 定位/车道线提取的横纵向噪声（合成出 lat_err 噪声 ≈ 0.02 m） |
| 位置 y | `vehicle.y` | 0.02 m | 同上 |
| 朝向 | `vehicle.yaw_deg` | 0.115° (≈0.002 rad) | yaw 估计误差 |
| 车速 | `vehicle.speed_kph` | 0.18 km/h (≈0.05 m/s) | 轮速/GNSS 速度噪声 |
| 横摆率 | `vehicle.yawrate` | 0.002 rad/s | IMU 陀螺噪声 |

x/y 都加 σ=0.02m 是因为 `lat_err = cos(θ)·dy − sin(θ)·dx` 在 cos/sin 平方和 = 1 的约束下，最终落到 lat_err 上的噪声 σ 仍 ≈ 0.02 m，与用户选定的"lat_err σ=0.02 m"一致。

**结构**：单步白噪声，每个 50Hz 步独立采样、每个 batch 元素独立采样，**3σ 截断**避免极端样本毒化梯度。

### 指令抖动（中档）

控制器算出 `delta_front / torque_wheel` 后，在送进 `vehicle.step` 之前加高斯抖动，模拟 ECU 量化与执行器小幅扰动。

| 通道 | σ | 截断 |
|------|---|------|
| `delta_front`（前轮转角，rad）| 0.001 rad (≈0.057°) | 3σ |
| `torque_wheel`（车轮总扭矩，N·m）| 15 N·m | 3σ |

**结构**：同样单步白噪声、每 batch 元素独立、3σ 截断。memoryless——本步抖动不影响下一步抖动采样。

### 关键不变量

- **真值用于 loss**：`run_simulation_batch` 末尾算 lateral_error / heading_error 时用 `vehicle.x / y / yaw`（真值），噪声只影响"控制器决策时的输入"和"执行器到达 plant 的指令"
- **history 记录控制器真实输出**：history 里 `steer / acc / torque` 是控制器算出来的值，不含 dither；vehicle.step 收到的是含 dither 版本
- **MLP 输入**：MLP 残差网络看到的 plant 输入是含 dither 的 `delta / torque`；这与训练时的 nominal 输入分布有偏差，但与"DR 把车辆参数推到 ±10%"是同一性质的失配，由使用方按场景决定 MLP 开关

## 实现路径

### 改动 1：`run_simulation_batch` 增加噪声/抖动注入

`sim/optim/train_batch.py` 的 `run_simulation_batch` 内部循环（当前 line 1130~1186 区段）：

```python
# 在 controller 调用之前
if noise_cfg is not None and noise_cfg['enable']:
    x_meas = vehicle.x + sample_clipped_normal(B, sigma_x, gen)
    y_meas = vehicle.y + sample_clipped_normal(B, sigma_y, gen)
    yaw_deg_meas = vehicle.yaw_deg + sample_clipped_normal(B, sigma_yaw_deg, gen)
    speed_kph_meas = vehicle.speed_kph + sample_clipped_normal(B, sigma_speed_kph, gen)
    yawrate_meas = vehicle.yawrate + sample_clipped_normal(B, sigma_yawrate, gen)
else:
    x_meas, y_meas = vehicle.x, vehicle.y
    yaw_deg_meas, speed_kph_meas = vehicle.yaw_deg, vehicle.speed_kph
    yawrate_meas = vehicle.yawrate

steer_out, ... = lat_ctrl.compute(x=x_meas, y=y_meas,
                                  yaw_deg=yaw_deg_meas,
                                  speed_kph=speed_kph_meas,
                                  yawrate=yawrate_meas, ...)
acc_cmd = lon_ctrl.compute(x=x_meas, y=y_meas,
                           yaw_deg=yaw_deg_meas,
                           speed_kph=speed_kph_meas, ...)
delta_front = steer_out / steer_ratio * DEG2RAD
torque_wheel = lon_ctrl.compute_torque_wheel(acc_cmd, vehicle.v, a_actual)

# 在 vehicle.step 之前
if dither_cfg is not None and dither_cfg['enable']:
    delta_to_plant = delta_front + sample_clipped_normal(B, sigma_delta, gen)
    torque_to_plant = torque_wheel + sample_clipped_normal(B, sigma_torque, gen)
else:
    delta_to_plant, torque_to_plant = delta_front, torque_wheel

# history 记录控制器真实输出
h_steer.append(steer_out)
h_torque.append(torque_wheel)
# ... 其它不变 ...

vehicle.step(delta=delta_to_plant, torque_wheel=torque_to_plant)
```

`sample_clipped_normal(B, sigma, gen)` 用 `torch.randn` 采样后用 `torch.clamp(_, -3σ, 3σ)`，复用 `torch.Generator` 锁种子。`requires_grad=False`——噪声是 detach 的常量加项，autograd 链路不受影响（梯度等于 1·∂loss/∂input，与无噪声时数值不同但拓扑相同）。

`hard_mode=True`（V1 验证）时**默认关闭噪声/抖动**——验证用真值评估；如果用户想看带噪部署性能，由 CLI 显式开启另一档。

### 改动 2：`default.yaml` 增加配置段

```yaml
feedback_noise:
  enable: false
  sigma_x_m: 0.02
  sigma_y_m: 0.02
  sigma_yaw_deg: 0.115
  sigma_speed_kph: 0.18
  sigma_yawrate_radps: 0.002
  clip_sigmas: 3.0

command_dither:
  enable: false
  sigma_delta_rad: 0.001
  sigma_torque_nm: 15.0
  clip_sigmas: 3.0

noise_seed: null  # null = 不锁种子；整数 = 可复现
```

### 改动 3：CLI flag

`train_batch.py` 增加：
- `--noise-enable / --no-noise`：状态反馈噪声总开关
- `--dither-enable / --no-dither`：指令抖动总开关
- `--noise-seed <int>`：噪声采样随机种子
- `--sigma-{x,y,yaw,speed,yawrate,delta,torque} <float>`：覆盖默认 σ（一次性实验用）

CLI 优先级 > yaml；不显式指定走 yaml 默认（`enable=false` 时与现有路径完全一致）。

### 改动 4：种子管理

`noise_seed` 与 `dr_seed` 解耦。`train_batch.train_batch()` 入口同时接收两个 seed，分别 `torch.Generator()`：
- `dr_gen`：每个 epoch 顶层采样 `(m_t, Cf, Cr)`（已有）
- `noise_gen`：仿真循环内部每步采状态噪声 + 指令抖动（新增）

这样可以复现"同一组车辆参数 + 不同噪声轨迹"或反之的对比实验。

## Loss 与训练流程

不变：
- 仍是 192 batch 元素的 average loss（K=4 domain × 48 traj，叠加噪声）
- per-trajectory baseline 软归一化（K=4 副本平均）依然有效——noise 让 baseline 估计噪声更大，但平均下来仍是无偏估计
- L2 正则、参数投影、grad_clip=10.0 不变

变化：
- 单 epoch 噪声调用次数：B=192 × T_max(~250) × 7 通道 ≈ **34 万次** `randn`，CPU 上 < 0.5s 开销，几乎不影响训练时长
- grad_norm 期望比无噪情况大 30~50%（噪声扰动让控制器输出更跳）。若实测 grad_norm 长期 >5、loss 不收敛，把 sigma 整体打 0.5×

### 训练打印增强

在现有 epoch 末尾打印基础上加：
- 当前 epoch 的噪声 σ 值（噪声/抖动开启时）
- 每个 domain 上去噪 lat_rmse（用真值算）
- 噪声 vs 真值的 lat_err 差距分布（mean abs 差），监控噪声实际幅度

## 验证流程

post_training 三组对比，全部用 V1 路径（`hard_mode=True`、噪声关）：

| 组 | 训练配置 | 验证场景 |
|----|---------|---------|
| baseline | default.yaml | 49 场景，无噪 |
| nominal-tuned | DR 关、噪声关训出来的 tuned yaml | 49 场景，无噪 |
| DR-tuned | DR 开、噪声关（即 2026-05-08 首跑那条线） | 49 场景，无噪 |
| **DR+noise-tuned**（本设计） | DR 开、噪声开训出来的 tuned yaml | 49 场景，无噪 |

**期望**：
- DR+noise-tuned 在无噪 49 场景上 lat_rmse 不显著退化于 DR-tuned（≤ 5%）
- 控制器参数（特别是 PID 增益）相对 DR-tuned 略保守——噪声训练通常逼出更小的 kp、更平滑的 T 段，避免对噪声过度反应

**附加验证（可选，非必跑）**：把 DR+noise-tuned 在"带噪 V1"路径上再跑一次（`hard_mode=True` + 噪声开），看带噪部署 lat_rmse 是否优于 DR-tuned 在带噪路径上的表现。这条不进默认 post_training，由用户手动跑 `validate_batch.py` 触发。

## 测试

### 单元测试（新增 `sim/tests/test_noise_dither.py`）

- `test_clipped_normal_shape_sigma`：1000 次采样，shape `[B]` / σ_emp 接近配置 σ / 无超出 ±3σ 的样本
- `test_noise_disabled_path_unchanged`：`enable=False` 时 `run_simulation_batch` 输出与现有路径**逐元素相等**（tolerance 0）
- `test_noise_seed_reproducible`：同 seed 跑两次，state proxy / dither 张量完全一致

### 回归测试

- `test_train_batch_noise_disabled_matches_existing`：`--noise-disable --no-dither` 跑 2 epoch，最终参数与现有 `train_batch.py` 结果完全一致

### 端到端（手动）

DR+noise 6 epoch（`--dr-enable --noise-enable --dither-enable --dr-seed 2026 --noise-seed 2026`）：
- loss 单调下降（允许某一 epoch 反弹，但末段相对首 epoch 改善）
- 无 NaN、参数无飞
- post_training 49 场景跑得通
- worst-domain loss 也下降

## 可微参数变化对照

无新增可微参数。控制器可微集（横向 T2/T3/T4/T6 + 纵向 7 PID + switch_speed）保持不变。噪声/抖动是 detach 的加项，不参与梯度。

## 后续可扩展

- 加入 per-domain 传感器零偏（每 domain 一个 bias 常数，整 epoch 不变；与本次单步白噪声叠加）
- 加入 OU 漂移（慢变 colored noise）
- 加入执行器一阶延迟 + rate-limit 抖动（需要给 vehicle.step 内部加 `_prev_delta_actual` 的有状态滤波，改动比本设计大）
- 把噪声 σ 也作为可调参数或 epoch-递增调度
- 切换到 minimax / CVaR loss 做 worst-case robust optimization
- 把噪声配置同步进 `dynamic / hybrid_v2` plant
