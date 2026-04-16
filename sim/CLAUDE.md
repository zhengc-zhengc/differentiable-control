# sim/ — 可微控制仿真

## 概述

将 C++ 控制器用 PyTorch 复现（`nn.Module`），搭配三种可选被控对象，通过 BPTT 进行梯度调参。
所有模块支持 `differentiable=True/False` 开关：`True` 全程 tensor 运算可回传梯度；`False` 与 V1（numpy）行为一致。

被控对象切换通过 `--plant` 参数或 `cfg['vehicle']['model_type']` 配置：
- `kinematic`：运动学自行车模型（默认，训练快）
- `dynamic`：6-DOF 动力学模型（RK4 积分）
- `hybrid_dynamic`：机理模型(Euler) + MLP 残差修正（逼近 CarSim 高保真仿真，需 plant 仓库 checkpoint）

## 模块结构

```
sim/
├── common.py              # 基础运算：lookup1d, smooth_clamp/sign, rate_limit (STE), PID, IIR
├── config.py              # YAML 加载/保存，table_from_config → (Tensor, Tensor)
├── sim_loop.py            # 50Hz 闭环仿真（lat→lon→vehicle step）
├── run_demo.py            # 可视化 Demo（--save --no-show --config）
├── compare_results.py     # [deprecated, 已被 post_training.py 取代] 旧版 4 场景对比
├── health_check.py        # 一键体检（测试 + 基线性能 + 梯度健康）
├── model/
│   ├── vehicle.py         # BicycleModel — 运动学模型 (x,y,yaw,v)，dt=0.02s
│   ├── dynamic_vehicle.py # DynamicVehicle — 6-DOF 动力学模型适配器，接口与 BicycleModel 一致
│   ├── hybrid_dynamic_vehicle.py # HybridDynamicVehicle — 机理模型+MLP 残差修正（plant 仓库集成）
│   ├── vehicle_factory.py # create_vehicle() — 根据 cfg 创建 kinematic/dynamic/hybrid_dynamic 模型
│   └── trajectory.py      # 轨迹生成（8 标准类型×6 速度段 + park_route）、变速剖面、expand_trajectories() + TrajectoryAnalyzer
├── controller/
│   ├── lat_truck.py       # LatControllerTruck (nn.Module, 可微:T2-T4/T6, 固定:kLh/T1/T5/T7/T8)
│   └── lon.py             # LonController (nn.Module, 可微:7 PID, 固定:L1-L5)
├── optim/
│   ├── train.py           # 可微调参（--trajectories 接收类型名自动展开全速度段、per-traj loss 归一化、TBPTT）
│   └── post_training.py   # 训练后自动化 + 独立验证 CLI（49 场景对比图、参数变化图、实验日志）
├── configs/
│   ├── default.yaml       # 默认参数（C++ 原始控制器参数，作为训练基线）
│   └── tuned/             # 调参结果 YAML（文件名含 commit hash + timestamp）
├── results/               # 结果图
│   ├── baseline/          # 基线结果（纳入 git，按被控对象分目录）
│   │   ├── kinematic/     # 运动学模型基线
│   │   ├── dynamic/       # 动力学模型基线
│   │   └── hybrid_dynamic/ # 混合模型基线
│   └── training/          # 训练产物（.gitignore 排除，按被控对象+时间戳分目录）
│       └── {plant}/{timestamp}/  # loss_curve.png, loss_breakdown.png, comparison_*.png, experiment_log.yaml
├── learn/                 # 学习笔记与调试日志（不影响运行）
└── tests/                 # pytest 测试
```

## 数据流

**仿真**：`trajectory` 生成参考轨迹 → `sim_loop` 每步调用 `lat_ctrl.compute()` → `lon_ctrl.compute()` → kinematic 直接 `vehicle.step(acc=...)`；dynamic/hybrid 追加 `lon_ctrl.compute_torque_wheel()` 后 `vehicle.step(torque_wheel=...)` → 记录 history

**训练**：`DiffControllerParams` 封装两个控制器（nn.Module） → `sim_loop(differentiable=True, tbptt_k=K)` 构建计算图（每 K 步 detach 状态截断梯度链） → `tracking_loss` 汇总误差 → per-trajectory loss 归一化（除以 epoch-1 baseline） → `backward()` → Adam + CosineAnnealingLR 更新 → 参数投影（非负约束） → `save_tuned_config` 导出 YAML

**loss 公式**：`loss = w_lat(10) × lat_mse + w_head(8) × head_mse + w_speed(3) × speed_mse + w_steer_rate(0.05) × Δsteer² + w_acc_rate(0.01) × Δacc² + λ(0.01) × Σ(p - p_init)²`。括号内为默认权重。L2 正则项惩罚参数偏离初始值。

**梯度路径**：loss → history 中的 tensor → vehicle 状态传播 → 控制器输出 → 查找表 y 值 / PID 增益（nn.Parameter）。不可微操作的处理：rate_limit 用 Straight-Through Estimator，条件分支用 smooth_step 混合，argmin 用 detach 隔离。T6 梯度通过 `query_by_time_differentiable` 回传（仅在变曲率轨迹上有梯度）。

**验证**：训练使用平滑近似（`differentiable=True`），得到的参数必须用原始硬限幅控制器（`differentiable=False`，即 V1 路径）重新跑仿真验证，确保参数在真实限幅逻辑下同样有效。验证覆盖 49 场景（8 类型 × 6 速度段 + park_route，速度 5~55 kph）。

## 常用命令

```bash
cd sim/
python -m pytest tests/ -q                                  # 跑测试
python health_check.py                                      # 一键体检（测试+基线+梯度）
python run_demo.py --save --no-show                         # 生成结果图（运动学模型）
python run_demo.py --plant dynamic --save --no-show         # 生成结果图（动力学模型）
python run_demo.py --plant hybrid_dynamic --save --no-show  # 生成结果图（混合模型，需 plant checkpoint）
python run_demo.py --config configs/tuned/xxx.yaml          # 加载调参结果
python optim/train.py --epochs 6                            # 全量训练（8 类型×6 速度段=48 条轨迹）
python optim/train.py --trajectories lane_change clothoid_decel  # 指定类型（自动展开到全速度段）
python optim/train.py --config configs/tuned/xxx.yaml --epochs 6  # warm-start 继续训练
python optim/train.py --plant hybrid_dynamic --epochs 6     # 指定被控对象
python optim/post_training.py --config configs/tuned/xxx.yaml              # 独立验证（全量 49 场景）
python optim/post_training.py --config configs/tuned/xxx.yaml --trajectories lane_change  # 指定类型验证
python optim/post_training.py --config configs/tuned/xxx.yaml --plant dynamic  # 指定被控对象验证
```

## train.py 参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 100 | 训练轮数（推荐 6，快速迭代） |
| `--lr` | 5e-2 | 主学习率（PID 增益） |
| `--lr-tables` | 5e-2 | 查找表 y 值学习率 |
| `--trajectories` | None（全量 8×6=48） | 轨迹类型名，自动展开到全速度段 |
| `--sim-length` | None | 仿真距离限制 (m) |
| `--tbptt-k` | 150 | TBPTT 截断窗口（步数，3 秒） |
| `--grad-clip` | 10.0 | 梯度范数裁剪阈值 |
| `--snapshot-interval` | 10 | 参数快照间隔（epoch） |
| `--plant` | None | 被控对象：kinematic / dynamic / hybrid_dynamic |
| `--config` | None | warm-start 配置文件路径 |
| `--w-lat/head/speed` | 10/8/3 | loss 权重（横向/航向/速度） |
| `--w-steer-rate/acc-rate` | 0.05/0.01 | 平滑度正则权重 |

## 与 controller_spec.md 的差异

- **符号约定**：spec 用 CW+（顺时针正），实现用 CCW+（逆时针正）。`lat_truck.py` Steps 4/5/7 的符号与 spec 相反，三处翻转自洽，最终输出数学等价
- **横向控制器**：spec §2.5 Steps 1-10 全部实现，无遗漏
- **纵向控制器**：spec §4.5 Steps 1-6 + IIR 滤波全部实现。Step 8 CalFinalTorque 部分实现（`compute_torque_wheel`：风阻/滚阻/惯性力/P补偿 → 车轮扭矩，跳过坡度估计和传动比/效率）。Steps 7/9（GearControl / 输出分配）跳过
- **纵向时间坐标**：spec 用 `absolute_time`，实现用 `relative_time`（仿真中等价）；速度误差用 1.0s 预瞄点（更具前瞻性）

## 控制器参数分类

> 变更可微参数范围后须同步更新此表。详见 `docs/tunable_params_analysis.md`。

### 可微参数（nn.Parameter，参与梯度优化）

| 控制器 | 参数 | 说明 |
|--------|------|------|
| 横向 LatControllerTruck | T2_y (prev_time_dist) | 预瞄距离时间系数 |
| | T3_y (reach_time_theta) | 收敛时间因子 |
| | T4_y (T_dt) | 角速度误差预瞄时间 |
| | T6_y (far_point_time) | 远预瞄点时间（前馈核心输入） |
| 纵向 LonController | station_kp, station_ki | 站位 PID 增益 |
| | low_speed_kp, low_speed_ki | 低速速度 PID 增益 |
| | high_speed_kp, high_speed_ki | 高速速度 PID 增益 |
| | switch_speed | 低/高速 PID 切换点 |

### 固定参数（buffer，不参与优化）

| 控制器 | 参数 | 固定理由 |
|--------|------|----------|
| 横向 | kLh | 车辆物理属性（铰接修正） |
| 横向 | T1_y (max_theta_deg) | 安全约束（航向误差上限） |
| 横向 | T5_y (near_point_time) | 不参与反馈控制律（仅输出监控） |
| 横向 | T7_y (max_steer_angle) | 物理极限（转向机构） |
| 横向 | T8_y (slip_param) | 车辆物理属性（轮胎侧滑） |
| 纵向 | L1_y (acc_up_lim) | 物理极限（动力系统能力） |
| 纵向 | L2_y (acc_low_lim) | 物理极限（制动系统能力） |
| 纵向 | L3_y (acc_up_rate) | 安全/舒适约束 |
| 纵向 | L4_y (acc_down_rate) | 安全/舒适约束 |
| 纵向 | L5_y (acc_rate_gain) | 安全约束辅助 |
| 横向 | rate_limit_fb/ff/total | 硬编码安全约束 |
| 纵向 | lon_torque 段 10 项（veh_mass/coef_cd/coef_rolling 等）| 物理常数（对齐 C++ CalFinalTorque） |

## 参考文档

- `docs/controller_spec.md` — 控制器完整规格（v1）
- `docs/controller_spec_v2.md` — 控制器规格 v2（对 v1 的校对修正，含纵向扭矩模型/坡度估计）
- `docs/tunable_params_analysis.md` — 可调参数分析
- `docs/plans/2026-03-02-differentiable-tuning-v2-design.md` — V2 设计
- `docs/plans/2026-04-15-torque-output-layer-design.md` — 扭矩输出层设计

## 训练规范

- **训练脚本必须实时打印每个 epoch 的进度**，包括：loss、各分项 RMSE（lat/head/speed）、梯度范数、耗时、NaN 梯度数
- 多轨迹时打印分轨迹 loss 明细（lat/head/speed RMSE + loss 分项）
- 每 N epoch（默认 10）打印参数快照：当前值、初始值、变化量、变化百分比
- 训练完成后打印汇总：初始 loss → 最终 loss（含变化量和百分比）、总耗时
- **训练完成后自动运行 `post_training`**：生成 loss 曲线、分轨迹 loss 分项图、baseline vs tuned 对比图（轨迹/横向误差/转向角/加速度）、实验日志 YAML，全部保存到 `results/training/{timestamp}/`

## 训练最佳实践

**默认训练集**（8 类型 × 6 速度段 = 48 条轨迹）：lane_change, double_lc, clothoid_left, clothoid_right, s_curve（恒速）+ combined_decel, clothoid_decel（弯前减速）+ lc_accel（梯形加减速），覆盖 5/18/25/35/45/55 kph 全速度段。

**推荐流程**：默认配置训练 6 epoch → 检查 V1 验证 → warm-start 继续训练 → 若 5kph 退化 > 15% 或 heading 退化 > 30% 则停止。

**参数投影约束**：T2/T3/T4/T6 时间参数 ≥ 0，所有 PID 增益 ≥ 0，switch_speed ∈ [0.5, 10]。

**已验证结论**：
- lr=0.05 最优（0.03 太慢，0.1 不稳定）
- TBPTT k=150 足够（k=0 full BPTT 无额外收益）
- combined_55kph 无法改善（控制器架构限制，非调参问题）
- warm-start 可进一步改善至 50-66%，但 heading 代价增大
- T4（角速度预瞄）持续降低 ~60% 是最大改善驱动因子

## 梯度爆炸防治（关键！）

**将硬限幅代码改为可微版本时，先化简数学表达式再选择近似方式**。例如 `sign(x)*min(|x|,L)` 本质是 `clamp(x,-L,L)`，直接用 STE clamp（导数=1）而非 smooth_sign（导数=100）。smooth 近似的 temperature 过小会导致 BPTT 链式乘法溢出（梯度爆炸≠loss 爆炸，用有限差分可验证真实梯度有界）。详见 `docs/bptt_gradient_explosion_analysis.md`。

## hybrid_dynamic 模型关键约束

- **必须用 Euler 积分**：MLP 训练时 base 用 Euler，用 RK4 会导致残差不匹配
- **参数与 dynamic 不同**：corner_stiff=56000, air_density=1.206（与 MLP 训练时一致，不同于 dynamic 的 80000/1.225）
- **MLP 冻结**：权重 requires_grad=False，但梯度通过计算图流到控制器参数
- **Checkpoint 路径**：`configs/checkpoints/best_error_model_from_carsim.pth`（相对于 sim/）
- **MLP 输入**：[state(6), delta_sw(1), T_rl(1), T_rr(1), dt(1)] = 10D，归一化后输入
- **MLP 输出**：3D [Δvx, Δvy, Δr] → 旋转到世界系 → ×dt 得位置修正 → 拼接为 6D

## 备注

- CPU only（单步计算量极小，GPU kernel 启动开销反而拖慢）
- 训练中用 `clip_grad_norm_` 作为梯度安全网（不应作为主要手段，优先调 temperature）
