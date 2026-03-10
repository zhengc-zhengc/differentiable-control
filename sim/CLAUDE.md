# sim/ — 可微控制仿真

## 概述

将 C++ 控制器用 PyTorch 复现（`nn.Module`），搭配运动学自行车模型，通过 BPTT 进行梯度调参。
所有模块支持 `differentiable=True/False` 开关：`True` 全程 tensor 运算可回传梯度；`False` 与 V1（numpy）行为一致。

## 模块结构

```
sim/
├── common.py              # 基础运算：lookup1d, smooth_clamp/sign, rate_limit (STE), PID, IIR
├── config.py              # YAML 加载/保存，table_from_config → (Tensor, Tensor)
├── sim_loop.py            # 50Hz 闭环仿真（lat→lon→vehicle step）
├── run_demo.py            # 可视化 Demo（--save --no-show --config）
├── compare_results.py     # 调参前后对比（轨迹 + 横向误差对比图）
├── health_check.py        # 一键体检（测试 + 基线性能 + 梯度健康）
├── model/
│   ├── vehicle.py         # BicycleModel — 运动学模型 (x,y,yaw,v)，dt=0.02s
│   ├── dynamic_vehicle.py # DynamicVehicle — 6-DOF 动力学模型适配器，接口与 BicycleModel 一致
│   ├── vehicle_factory.py # create_vehicle() — 根据 cfg 创建 kinematic/dynamic 模型
│   └── trajectory.py      # 5 种轨迹生成 + TrajectoryAnalyzer（detached argmin）
├── controller/
│   ├── lat_truck.py       # LatControllerTruck (nn.Module, 可微:T2-T6, 固定:kLh/T1/T7/T8)
│   └── lon.py             # LonController (nn.Module, 可微:7 PID, 固定:L1-L5)
├── optim/
│   ├── train.py           # 可微调参（参数快照、分轨迹 loss、完整训练历史）
│   └── post_training.py   # 训练后自动化（loss 曲线、对比图、实验日志）
├── configs/
│   ├── default.yaml       # 默认参数（C++ 原始控制器参数，作为训练基线）
│   └── tuned/             # 调参结果 YAML（文件名含 commit hash + timestamp）
├── results/               # 结果图（纳入 git）
│   ├── baseline/          # 基线结果（用于调参前后对比）
│   └── training/          # 训练产物（每次训练一个时间戳子目录）
│       └── {timestamp}/   # loss_curve.png, loss_breakdown.png, comparison_*.png, experiment_log.yaml
├── learn/                 # 学习笔记与调试日志（不影响运行）
└── tests/                 # pytest 测试
```

## 数据流

**仿真**：`trajectory` 生成参考轨迹 → `sim_loop` 每步调用 `lat_ctrl.compute()` → `lon_ctrl.compute()` → `vehicle.step()` → 记录 history

**训练**：`DiffControllerParams` 封装两个控制器（nn.Module） → `sim_loop(differentiable=True)` 构建计算图 → `tracking_loss` 汇总误差（横向/航向/速度/平滑度） → `backward()` → Adam 更新 nn.Parameter → `save_tuned_config` 导出 YAML

**梯度路径**：loss → history 中的 tensor → vehicle 状态传播 → 控制器输出 → 查找表 y 值 / PID 增益（nn.Parameter）。不可微操作的处理：rate_limit 用 Straight-Through Estimator，条件分支用 smooth_step 混合，argmin 用 detach 隔离。

**验证**：训练使用平滑近似（`differentiable=True`），得到的参数必须用原始硬限幅控制器（`differentiable=False`，即 V1 路径）重新跑仿真验证，确保参数在真实限幅逻辑下同样有效。`run_demo.py --config` 默认即为 V1 路径。

## 常用命令

```bash
cd sim/
python -m pytest tests/ -q                                  # 跑测试
python health_check.py                                      # 一键体检（测试+基线+梯度）
python run_demo.py --save --no-show                         # 生成结果图（运动学模型）
python run_demo.py --plant dynamic --save --no-show         # 生成结果图（动力学模型）
python run_demo.py --config configs/tuned/xxx.yaml          # 加载调参结果
python optim/train.py --epochs 50 --trajectories circle sine  # 训练（运动学模型）
python optim/train.py --plant dynamic --epochs 50           # 训练（动力学模型）
```

## 与 controller_spec.md 的差异

- **符号约定**：spec 用 CW+（顺时针正），实现用 CCW+（逆时针正）。`lat_truck.py` Steps 4/5/7 的符号与 spec 相反，三处翻转自洽，最终输出数学等价
- **横向控制器**：spec §2.5 Steps 1-10 全部实现，无遗漏
- **纵向控制器**：spec §4.5 Steps 1-6 + IIR 滤波全部实现；Steps 7-9（GearControl / CalFinalTorque / 输出分配）有意跳过，直接输出加速度 (m/s²) 而非扭矩/刹车/档位
- **纵向时间坐标**：spec 用 `absolute_time`，实现用 `relative_time`（仿真中等价）；速度误差用 1.0s 预瞄点（更具前瞻性）

## 控制器参数分类

> 变更可微参数范围后须同步更新此表。详见 `docs/tunable_params_analysis.md`。

### 可微参数（nn.Parameter，参与梯度优化）

| 控制器 | 参数 | 说明 |
|--------|------|------|
| 横向 LatControllerTruck | T2_y (prev_time_dist) | 预瞄距离时间系数 |
| | T3_y (reach_time_theta) | 收敛时间因子 |
| | T4_y (T_dt) | 角速度误差预瞄时间 |
| | T5_y (near_point_time) | 近预瞄点时间 |
| | T6_y (far_point_time) | 远预瞄点时间 |
| 纵向 LonController | station_kp, station_ki | 站位 PID 增益 |
| | low_speed_kp, low_speed_ki | 低速速度 PID 增益 |
| | high_speed_kp, high_speed_ki | 高速速度 PID 增益 |
| | switch_speed | 低/高速 PID 切换点 |

### 固定参数（buffer，不参与优化）

| 控制器 | 参数 | 固定理由 |
|--------|------|----------|
| 横向 | kLh | 车辆物理属性（铰接修正） |
| 横向 | T1_y (max_theta_deg) | 安全约束（航向误差上限） |
| 横向 | T7_y (max_steer_angle) | 物理极限（转向机构） |
| 横向 | T8_y (slip_param) | 车辆物理属性（轮胎侧滑） |
| 纵向 | L1_y (acc_up_lim) | 物理极限（动力系统能力） |
| 纵向 | L2_y (acc_low_lim) | 物理极限（制动系统能力） |
| 纵向 | L3_y (acc_up_rate) | 安全/舒适约束 |
| 纵向 | L4_y (acc_down_rate) | 安全/舒适约束 |
| 纵向 | L5_y (acc_rate_gain) | 安全约束辅助 |
| 横向 | rate_limit_fb/ff/total | 硬编码安全约束 |

## 参考文档

- `docs/controller_spec.md` — 控制器完整规格
- `docs/tunable_params_analysis.md` — 可调参数分析
- `docs/plans/2026-03-02-differentiable-tuning-v2-design.md` — V2 设计

## 训练规范

- **训练脚本必须实时打印每个 epoch 的进度**，包括：loss、各分项 RMSE（lat/head/speed）、梯度范数、耗时、NaN 梯度数
- 多轨迹时打印分轨迹 loss 明细（lat/head/speed RMSE + loss 分项）
- 每 N epoch（默认 10）打印参数快照：当前值、初始值、变化量、变化百分比
- 训练完成后打印汇总：初始 loss → 最终 loss（含变化量和百分比）、总耗时
- **训练完成后自动运行 `post_training`**：生成 loss 曲线、分轨迹 loss 分项图、baseline vs tuned 对比图（轨迹/横向误差/转向角/加速度）、实验日志 YAML，全部保存到 `results/training/{timestamp}/`

## 梯度爆炸防治（关键！）

**将硬限幅代码改为可微版本时，先化简数学表达式再选择近似方式**。例如 `sign(x)*min(|x|,L)` 本质是 `clamp(x,-L,L)`，直接用 STE clamp（导数=1）而非 smooth_sign（导数=100）。smooth 近似的 temperature 过小会导致 BPTT 链式乘法溢出（梯度爆炸≠loss 爆炸，用有限差分可验证真实梯度有界）。详见 `docs/bptt_gradient_explosion_analysis.md`。

## 备注

- CPU only（单步计算量极小，GPU kernel 启动开销反而拖慢）
- 训练中用 `clip_grad_norm_` 作为梯度安全网（不应作为主要手段，优先调 temperature）
