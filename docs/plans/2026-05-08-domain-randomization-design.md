# 车辆物理参数域随机化设计

**日期**：2026-05-08
**状态**：设计完成，待实施
**范围**：truck_trailer plant，纯机理 base，无挂车，无 MLP

## 背景

当前可微调参流程把 plant 物理参数（牵引车质量 `m_t`、前后轴侧偏刚度 `Cf/Cr`）当作精确已知量，所有训练轨迹都跑在同一组 nominal 值上。但实车部署时这些量存在不可忽视的不确定性：

- `m_t` 受装载量、燃油量、驾驶员重量影响，相对空载真值会有几百 kg 到几吨的偏移
- `Cf`、`Cr` 是路面摩擦、轮胎温度、磨损共同决定的等效线性化值，实车标定本身就有误差，工况切换还会进一步放大偏离

不做随机化的训练相当于让控制器对 nominal 点 overfit。一旦实车偏离这个点，tuned 出来的参数可能不再是局部最优，甚至比默认参数退化。域随机化的目的是让训练过程中控制器看到一族 plant 参数的样本，得到对参数偏差更鲁棒的控制器参数。

## 目标

在 `train_batch.py` 的并行训练管线里增加一层域随机化：每个 epoch 开始时随机采样 K=4 组 `(m_t, Cf, Cr)`，48 条轨迹各复制 K 份对应这 K 个 domain，batch 维度从 48 扩到 192。控制器看到不同 domain 上的状态轨迹，loss 在 192 个仿真上累加再反传。

随机化范围采保守档：

| 参数 | nominal | 范围 | 分布 |
|---|---|---|---|
| `m_t` | 9300 kg | ±10%（8370~10230） | 均匀 |
| `Cf` | 264000 N/rad | ±20%（211200~316800） | 均匀 |
| `Cr` | 335000 N/rad | ±20%（268000~402000） | 均匀 |
| `Iz_t` | 48639 kg·m² | 由 `m_t` 联动：`Iz_t = Iz_t_nominal × (m_t / m_t_nominal)` | 不独立采样 |

`Iz_t` 与 `m_t` 联动是为了避免出现"轻车但转动惯量大"这种不物理的 domain。

## 非目标

- 不随机化挂车质量、几何参数、阻力系数、转向比、风阻面积
- 不在代码侧自动处理 DR × MLP 耦合（MLP 是按 nominal 车辆参数训练的，DR 把车辆参数推到 ±10/20% 区间时输入分布偏离训练域）；是否启用 MLP 由使用方按场景在 yaml / CLI 决定，2026-05-08 首跑选择关 MLP 是出于这一权衡，但代码不做强制
- 不在 kinematic、dynamic、hybrid_dynamic、hybrid_v2 这几条 plant 路径上启用 DR（仅 truck_trailer）
- 不在 scalar `train.py` 上做 DR（仅扩展 `train_batch.py`）
- 不做控制器参数的 robust optimization（如 minimax、CVaR），只做最朴素的 ensemble average loss
- 不引入传感器噪声、执行器扰动

## 域定义

### 采样时机

每个 epoch 开始前采样 K=4 组 `(m_t, Cf, Cr)`，整个 epoch 内 192 个 batch 元素的 domain 固定不变。下一个 epoch 重新采样新的 4 组。

**为什么不每个 step 重采**：每个 batch 元素持有的 plant 状态在 50Hz 时间轴上必须连续演化，中途换 domain 会让动力学跳变。

**为什么不 192 个全独立采样**：① 同 domain 跨轨迹可比，便于事后分析"哪个 domain 最难收敛"；② 采样数从 192 降到 4，loss 估计的方差更可控；③ Per-trajectory baseline 归一化更稳定（K=4 个副本的 baseline 取均值）。

### Batch 维度排列

192 个 batch 元素按 `[traj_idx, domain_idx]` 二维排列，flatten 顺序：

```
B[0..47]:    48 traj × domain_0
B[48..95]:   48 traj × domain_1
B[96..143]:  48 traj × domain_2
B[144..191]: 48 traj × domain_3
```

参数张量按 batch 维广播：`m_t.shape = [192]`，前 48 个元素是 domain_0 的 m_t，后 48 个是 domain_1，依此类推。

### 联动公式

```python
m_t_sample, Cf_sample, Cr_sample = sample_uniform(K=4)  # shape: [4]
Iz_t_sample = Iz_t_nominal * (m_t_sample / m_t_nominal)  # shape: [4]

# 广播到 192 维
m_t_batch = m_t_sample.repeat_interleave(48)  # shape: [192]
# 同理 Iz_t_batch / Cf_batch / Cr_batch
```

## 实现路径

四个改动点。

### 改动 1：`TruckTrailerNominalDynamics` 支持运行时切换 batched 参数

`sim/model/truck_trailer_dynamics.py` 当前把 `m_t / Iz_t / Cf / Cr` 等存为 scalar buffer。改造原则：

- **构造函数保持现状**（向后兼容）：依然接受 scalar `params` dict，初始化为 0-d buffer
- **新增 `set_domain(m_t, Cf, Cr)` 方法**：运行时把 `m_t / Cf / Cr` 替换为 shape `[B]` tensor，`Iz_t` 由公式联动算出。原有 0-d buffer 被覆盖为 `[B]` 张量
- **新增 `register_buffer` 保存 nominal 值**：`_m_t_nominal`、`_Iz_t_nominal`，专门用于 `set_domain` 内的联动计算

```python
def set_domain(self, m_t: Tensor, Cf: Tensor, Cr: Tensor):
    """运行时注入域参数。三者 shape 必须一致（[B] 或 0-d）。Iz_t 由 m_t 联动算出。"""
    self.m_t = m_t
    self.Iz_t = self._Iz_t_nominal * (m_t / self._m_t_nominal)
    self.Cf = Cf
    self.Cr = Cr
```

底层动力学算式（轮胎力、横摆力矩、铰接弹簧）天然 element-wise，自动广播到状态量 `[B, state_dim]`，不需要改。需要小心的是 `[B]` 张量与 `[B, state_dim]` 状态相乘时，要确保广播成 `[B, state_dim]` 而不是错误的 `[B, B]`——必要时用 `m_t.unsqueeze(-1)` 显式对齐。

DR 关闭时不调用 `set_domain`，原 scalar 路径完全不受影响。

### 改动 2：`BatchedTruckTrailerVehicle` 增加 domain 注入接口

`sim/model/truck_trailer_vehicle.py`（或 `train_batch.py` 内的 batched 适配器）增加 `set_domain(m_t, Cf, Cr)` 方法，内部调用底层 `nominal_dynamics.set_domain`。

调用时机：每个 epoch 开始时由训练循环调一次。

### 改动 3：`train_batch.py` 增加 DR 配置和采样逻辑

**配置文件优先，CLI 仅作覆盖**。`default.yaml` 新增段：

```yaml
domain_randomization:
  enable: false       # 总开关，false 走原路径完全不变
  K: 4                # 每个 epoch 采样的 domain 数
  mt_range: 0.10      # m_t 相对 nominal 的 ±range
  cfcr_range: 0.20    # Cf/Cr 相对 nominal 的 ±range
```

CLI 同名参数（`--dr-enable / --dr-K / --dr-mt-range / --dr-cfcr-range`）仅在显式指定时覆盖 cfg 值。这样切换 DR 既可以通过改 yaml（warm-start 场景）也可以通过 CLI（一次性实验）。

训练循环改造：

```python
# 在 epoch 循环顶部
if dr_enable:
    domains = sample_domains(K=dr_K, mt_range=..., cfcr_range=...)  # [K, 3]
    m_t_batch = domains[:, 0].repeat_interleave(N_traj)  # [K*N_traj]
    Cf_batch = domains[:, 1].repeat_interleave(N_traj)
    Cr_batch = domains[:, 2].repeat_interleave(N_traj)
    vehicle.set_domain(m_t_batch, Cf_batch, Cr_batch)

    # 轨迹张量复制 K 份
    expanded_traj_tensors = duplicate_trajectories(traj_tensors, K=dr_K)
else:
    # 原路径不变
    ...
```

`run_simulation_batch` 不需要改，它只看 `B` 维度的大小，不关心 batch 来源是同 traj 不同 domain 还是同 domain 不同 traj。

### 改动 4：MLP 关闭

复用 `validate_batch.py` 里 `_apply_config_overrides` 的机制：把 `cfg['truck_trailer_vehicle']['checkpoint_path']` 置空字符串，vehicle factory 会传 `None` 给 vehicle，MLP 不加载。

在 `train_batch.py` 加 `--disable-mlp` flag，触发同样的 cfg 改写。**`--dr-enable` 与 MLP 开关解耦**：是否启用 MLP 仅由 `cfg['truck_trailer_vehicle']['checkpoint_path']`（空串=关）和 `--disable-mlp` 入参决定，DR 不再强制覆盖。注意 MLP 是按 nominal 车辆参数训练的，DR 把车辆参数推到 ±10/20% 区间时 MLP 输入分布偏移训练域、残差解释力下降——这个事实由使用方按场景权衡，需要时显式传 `--disable-mlp` 即可。

把 `_apply_config_overrides` 抽到 `sim/config.py` 作为公共函数，validate_batch.py 和 train_batch.py 共用。

## Loss 归一化与训练流程

### Per-trajectory baseline

保持现状的 per-traj 软归一化，但在 K 个 domain 上做平均。利用前述固定 batch 布局（traj_idx = i % 48，domain_idx = i // 48），traj_key `j` 的 K 个副本索引就是 `[j, j+48, j+96, j+144]`：

```python
# Epoch 1：跑完后记 baseline（每条 traj 取 K 个副本的 loss 均值）
for j, traj_key in enumerate(traj_keys):  # j ∈ [0, 48)
    indices = [j + k * 48 for k in range(K)]  # K 个副本
    baseline_traj_losses[traj_key] = mean(per_element_losses[indices])

# Epoch 2+：用这个 baseline 归一化
for i in range(B):  # B = 192
    j = i % 48
    norm_factor = max(baseline_traj_losses[traj_keys[j]] ** alpha, norm_floor)
    weighted_loss[i] = per_element_loss[i] / norm_factor / B
```

`per_element_loss` 由现有 `batched_tracking_loss` 在 192 个仿真上算出（lat/head/speed/平滑度的逐元素加权和）。

### L2 正则

不变。L2 只惩罚控制器 `nn.Parameter`，plant 参数是 buffer 不参与。

### 梯度路径

控制器参数 → 输出 `delta`、`torque` → batched plant `step()` → batched 状态 history → loss。Plant 参数（`m_t / Cf / Cr / Iz_t`）作为常量插在算式里，梯度路径完全保留。这件事的健康检查由现有的梯度健康脚本承担（应该没有破坏链路）。

### 训练时打印增强

每个 epoch 末尾增加打印：
- 当前 epoch 的 4 个 domain 具体参数值
- 每个 domain 上的平均 loss（`mean over 48 traj × 1 domain`）
- worst domain（loss 最大的那一个）的 ID 和 loss

这能让人在训练过程中直观看到"哪一类 domain 让控制器最难收敛"。

## 验证流程

训练完成后，post_training 增加 **DR robustness grid 评估**。在 3×3×3 = 27 个 `(m_t, Cf, Cr)` 网格点上跑 V1 验证（关 MLP）：

```
m_t  ∈ {0.9×, 1.0×, 1.1×} × m_t_nominal
Cf   ∈ {0.8×, 1.0×, 1.2×} × Cf_nominal
Cr   ∈ {0.8×, 1.0×, 1.2×} × Cr_nominal
```

每个网格点跑 48 条轨迹，统计：
- 平均 RMSE（lat / head / speed）
- worst-case RMSE（最差 traj × domain 组合的指标）
- worst domain（在哪一个 grid 点上整体表现最差）

输出三组对比：
1. **默认 baseline**：未 tune 的 default.yaml
2. **Nominal-tuned**：DR 关闭、用现有 train_batch 跑出来的 tuned 参数
3. **DR-tuned**：DR 开启跑出来的 tuned 参数

期望结果：
- DR-tuned 在 nominal 点（1.0×, 1.0×, 1.0×）上的指标 ≈ Nominal-tuned（不显著退化）
- DR-tuned 的 worst-case RMSE 显著优于 Nominal-tuned
- 平均 RMSE 两者接近，DR-tuned 略差也是可接受代价

如果 DR-tuned 在 nominal 点退化超过 5% 而 worst-case 改善不明显，说明 DR 范围太大或 K 太小，需要重新配置。

## 测试

### 单元测试（新增）

`sim/tests/test_domain_randomization.py`：
- `test_nominal_dynamics_accepts_batched_params`：用 shape `[4]` 的 m_t/Cf/Cr 构造 dynamics，跑一步，输出 shape `[4, state_dim]` 且每个元素与单独 scalar 运算结果一致
- `test_set_domain_updates_iz`：调用 `set_domain` 后 `Iz_t` 按公式联动
- `test_sample_domains_in_range`：采样 1000 次，验证 m_t/Cf/Cr 落在配置区间内、分布近似均匀
- `test_train_batch_dr_enable_changes_params`：DR 开启训练 2 epoch，控制器参数应有非零变化

### 回归测试

`test_train_batch_dr_disabled_matches_existing`：DR 关闭时 (`--dr-enable=false`) 跑 2 epoch，最终参数与现有 train_batch 结果完全一致（数值 tolerance ~1e-6）。这条测试保护"DR 不影响默认路径"。

### 端到端测试

DR 开启 3 epoch（手动跑，不进 pytest）：
- loss 单调下降
- 参数没飞（无 NaN、无超出投影约束）
- post_training 验证 grid 跑得通
- worst domain loss 也在下降

## 可微参数变化对照

无新增可微参数。Plant 参数仍是 buffer，不参与梯度优化。控制器可微参数集（横向 T2/T3/T4/T6 + 纵向 7 PID + switch_speed）保持不变。

## 训练结果（2026-05-08，commit c446712）

### 训练配置

- 命令：`python -u optim/train_batch.py --plant truck_trailer --dr-enable --epochs 6 --dr-seed 2026`
- DR 开关：`enable=True`，K=4，m_t±10%，Cf/Cr±20%，dr_seed=2026（可复现）
- 本次首跑由"DR 强制 disable_mlp"的旧逻辑接管，因此训练与 post_training V1 对比都走纯机理 base 路径（baseline default.yaml 与 tuned 同条件，对比公平）。该强制耦合在 2026-05-09 已解除，是否启用 MLP 改由 yaml/CLI 自决；如需复现本次效果，命令应改为 `--dr-enable --disable-mlp ...`
- 训练集：48 条标准轨迹（8 类 × 6 速度）× K=4 → batch=192
- 总耗时：2866s（~48 min），其中训练 ~46 min + V1 对比 ~2 min

### Loss 收敛

| Epoch | Loss   | lat_rmse | head_rmse | spd_rmse | grad_norm |
|------:|-------:|---------:|----------:|---------:|----------:|
| 1 | 2.7083 | 0.7019m | 0.0252rad | 0.3290m/s | 1.59 |
| 2 | 1.6789 | 0.5734m | 0.0190rad | 0.3287m/s | 0.65 |
| 3 | 1.9756 | 0.6139m | 0.0230rad | 0.3257m/s | 1.07 |
| 4 | 1.5953 | 0.5470m | 0.0201rad | 0.3220m/s | 0.30 |
| 5 | 1.9181 | 0.6022m | 0.0220rad | 0.3199m/s | 1.04 |
| 6 | **1.5639** | 0.5489m | 0.0176rad | 0.3201m/s | 0.40 |

最终 loss 比初始下降 **42.3%**（2.7083 → 1.5639）。

### 49 场景 V1 对比（baseline vs DR-tuned，均无 MLP）

47/49 场景 lat_rmse 改善，集中在 25-48% 区间。两个轻微退化场景：

| 场景 | baseline lat_rmse | tuned lat_rmse | 变化 |
|------|------------------:|---------------:|-----:|
| S 弯 5kph | 0.0179 m | 0.0200 m | +12.0% |
| 组合弯 5kph | 0.0188 m | 0.0261 m | +38.8% |

退化幅度都在毫米级（< 1 cm），且仅出现在 5kph 极低速段。

显著改善示例：

| 场景 | baseline lat_rmse | tuned lat_rmse | 变化 |
|------|------------------:|---------------:|-----:|
| clothoid_decel 25kph | 0.8608 m | 0.4436 m | -48.5% |
| 组合弯 45kph | 1.3089 m | 0.6802 m | -48.0% |
| clothoid_decel 18kph | 0.3661 m | 0.2021 m | -44.8% |
| clothoid_decel 35kph | 1.8768 m | 1.0406 m | -44.6% |
| clothoid_decel 45kph | 2.8312 m | 1.6611 m | -41.3% |

heading_rmse 同步改善 15-40%，无显著退化。

### 关键参数变化

| 参数 | 初始 | 最终 | Δ |
|------|-----:|-----:|---|
| station_kp | 0.25 | 0.0623 | -75.1% |
| high_speed_ki | 0.01 | 0.1562 | +1462% |
| high_speed_kp | 0.34 | 0.5059 | +48.8% |
| low_speed_ki | 0.01 | 0.0137 | +36.6% |
| low_speed_kp | 0.35 | 0.3368 | -3.8% |
| switch_speed | 3.0 | 2.9747 | -0.8% |
| T2/T3/T4/T6_y | 见 tuned yaml | max\|Δ\|≈0.18 |  |

high_speed_ki 涨幅最大（积分项主导高速段稳态误差消除）；station_kp 大幅下降（DR 训练时质量扰动让大 kp 易抖动）；T2/T3/T4/T6 各速度段都有量级 ~0.18 的非零调整。

### 产物

- 调参 yaml：`sim/configs/tuned/tuned_c446712_20260508_133108.yaml`
- 训练曲线 + 49 场景对比图：`sim/results/training/truck_trailer/20260508_133208/`（gitignore 排除）
- 实验日志：同目录 `experiment_log.yaml`

## 后续可扩展（不在本设计内）

- 扩范围至 `±20%` m_t / `±30%` Cf-Cr（中等档）或更激进
- 加入挂车质量随机化（需配合 trailer_mask 离散切换 + MLP 残差处理）
- 加入几何参数（`a_t`、`L_t`）和阻力（`CdA`、`rolling_coeff`）
- MLP 残差与 base 参数解耦：训练新一版 MLP，把 `(m_t, Cf, Cr)` 也作为输入特征
- 切换到 robust optimization 目标（CVaR / minimax）而非 average loss
- DR 应用到 dynamic / hybrid_dynamic plant
- 引入传感器噪声、执行器延迟、rate limit 抖动
