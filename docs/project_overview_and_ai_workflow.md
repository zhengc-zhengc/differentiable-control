# 可微控制 sim 项目:技术总览 + AI 协作开发实录

本文档分两大部分。**第一部分**说明 `sim/` 目录里的可微控制仿真是怎么回事 —— 控制器架构、输入输出、怎么做到端到端可微、怎么用 BPTT 调参。**第二部分**说这个项目本身是怎么靠"人指挥 AI、AI 执行"的模式写出来的 —— 写代码的分工,出问题时怎么排查,以及为什么要主动给代码加"可观测性"(让 AI 后面更容易帮你分析)。

---

# Part I — 可微控制技术架构

## 1. 项目背景

这个仓库做两件事:

1. **控制器提取**:从一段自动驾驶控制模块的屏幕录像中,逐帧恢复 C++ 源码。清稿产物在 `src/mlp/control/`,对应两个控制器:`lat_controller_truck.cc`(重卡横向)、`lon_controller.cc`(纵向)。

2. **可微复现 + 调参**:把上面的 C++ 控制器用 PyTorch 重写(`sim/controller/`),接上一个车辆模型做 50Hz 闭环仿真,然后利用 PyTorch 的自动求导,把控制器里那些常数参数(查找表 y 值、PID 增益、预瞄时间)当成 `nn.Parameter`,**用梯度下降直接调**。目标是让仿真轨迹误差最小。

全程不用 RL、不用黑盒搜索,**就是把"控制器 + 车辆"拼成一张 50Hz × N 秒的计算图,反向传播**。

## 2. 总体数据流(50Hz 闭环)

```
                    ┌─────────────────────────────────────┐
                    │  参考轨迹 (x_ref, y_ref, v_ref, ...) │
                    └──────────────┬──────────────────────┘
                                   │ 最近点投影
                                   ▼
┌────────┐         ┌────────────────────────┐
│ Plant  │─state──>│   横向控制器 LatTruck   │────┐
│(bicycle│         │   (nn.Module,可微)     │    │ steer_sw(deg)
│ /6-DOF │         └────────────────────────┘    │
│ /truck │                                        ▼
│ trailer│         ┌────────────────────────┐   ┌───┐
│  +MLP) │<──────┤│   纵向控制器 LonController│◄──┤输出│
│        │  control│   (nn.Module,可微)     │   └───┘
│        │         └────────────────────────┘
└────────┘              │ acc_cmd (m/s²)
    ▲                   │ + compute_torque_wheel
    │                   │
    └───────────────────┘
         delta, torque_wheel / acc
```

每个控制周期:
1. plant 返回状态 `(x, y, yaw, v, yawrate)`
2. 横向控制器按轨迹最近点 + 远近预瞄点,算出**方向盘转角**
3. 纵向控制器按 Frenet 站位误差 + 预瞄速度,算出**加速度指令**
4. 运动学 plant 直接吞加速度;动力学 plant 先把加速度转成车轮扭矩再吞

## 3. 模块拓扑

```
sim/
├── common.py               # lookup1d / smooth_clamp / STE clamp / PID / IIR
├── config.py               # YAML 加载,表 → (Tensor, Tensor)
├── sim_loop.py             # 50Hz 闭环 + Truncated BPTT
├── controller/
│   ├── lat_truck.py        # 重卡横向(10 个 step,含 kLh 铰接修正)
│   └── lon.py              # 纵向(6 step + IIR 滤波 + 扭矩输出层)
├── model/
│   ├── vehicle.py          # kinematic 自行车
│   ├── dynamic_vehicle.py  # 6-DOF 单轨(乘用车)
│   ├── hybrid_dynamic_vehicle.py  # Euler base + MLP 残差
│   ├── truck_trailer_vehicle.py   # 牵引车+挂车双体(12D)+ 可选 MLP
│   └── vehicle_factory.py  # 按 cfg['vehicle']['model_type'] 分发
└── optim/
    ├── train.py            # scalar 训练(每场景串行,单 epoch ~40 min)
    ├── train_batch.py      # batched 并行训练(所有场景同步推进,单 epoch ~5 min)
    ├── validate_batch.py   # 自定义 A/B 对比验证
    └── post_training.py    # 训练后自动化验证 + 49 场景对比图
```

## 4. 控制器架构细节

### 4.1 横向 — LatControllerTruck(重卡,10 步)

按 `docs/controller_spec_v2.md §2.5` 的 10 步:

| 步 | 做什么 | 关键输入 |
|---|---|---|
| 1 | 查 8 张按速度索引的表(T1-T8),得到 `max_theta / prev_time / reach_time / T_dt / near/far_time / max_steer / slip_param` | `speed_kph` |
| 2 | 轨迹 3 点查询:最近点 + 近预瞄 + 远预瞄 | `(x, y)`, `t_now + near_time/far_time` |
| 3 | 计算横向误差、航向误差、远点曲率 | 最近点几何 |
| 4 | `real_theta = heading_error + atan(kLh·yawrate/v)` | 铰接修正(kLh=0 退化) |
| 5 | `real_dt_theta = yawrate − κ_far · v` | 角速度误差 |
| 6 | `target_theta = clamp(atan(−lat_err/prev_dist), ±max_theta)` | 横偏→误差角 |
| 7 | `target_κ = ((target_theta − real_theta) + (target_dt − real_dt)·T_dt)/denom` | 控制律 |
| 8 | 反馈转角 = `atan(target_κ·L)·RAD2DEG·steer_ratio·slip_param` → rate_limit_fb | 方向盘角 |
| 9 | 前馈转角 = `atan(κ_far·L)·RAD2DEG·steer_ratio·slip_param` → rate_limit_ff | 前瞻 |
| 10 | 合并 fb+ff → clamp → rate_limit_total | 输出方向盘角 |

**可微参数**:T2, T3, T4, T6 四张查找表的 y 值(每张 7 个点,共 28 个标量)。
**固定参数**:T1, T5, T7, T8(物理约束),kLh,三个 rate_limit 硬编码常数。

### 4.2 纵向 — LonController(6 步 + IIR + 扭矩输出)

| 步 | 做什么 |
|---|---|
| 1 | Frenet 投影:算站位误差 `s_ref − s_match` |
| 2 | 站位保护:高速(>1 kph)直通,低速按分支处理 |
| 3 | 站位 PID → `speed_offset` |
| 4 | 速度 PID(低/高速分段增益):输入 = `speed_offset + preview_speed_error`,输出 `acc_closeloop` |
| 5 | 前馈叠加 `acc_cmd = acc_closeloop + acc_use_preview_a · preview_accel_ref` |
| 6 | CalFinalAcc:L1/L2 幅值限,L3/L4 速率限,L5 速率增益,低速蠕行保护 |
| — | IIR(α=0.15)低通 |
| 扭矩 | `compute_torque_wheel(acc_cmd, v, a_actual)`:风阻 + 滚阻 + 惯性 + P 补偿 → T_wheel |

**可微参数**:4 组 PID 增益(7 个标量)+ switch_speed。
**固定参数**:L1-L5 查找表(物理限制),扭矩层 10 个物理常数(veh_mass / coef_cd / 等)。

## 5. 可微化关键技术

这是整个项目的核心挑战。原始 C++ 控制器里到处是硬分支(`if speed > threshold`)和硬限幅(`clamp`),梯度传不过去。

### 5.1 技术清单

| 非可微原语 | 可微替代 | 代码位置 |
|---|---|---|
| `clamp(x, lo, hi)` | **Straight-Through Estimator**:前向硬 clamp,反向导数=1 | `common._straight_through_clamp` |
| `min(x, lo) / max(x, hi)`(单侧) | `smooth_lower_bound / smooth_upper_bound`(softplus) | `common.smooth_lower_bound` |
| `sign(x) · min(\|x\|, L)` | 化简后直接等于 `clamp(x, -L, L)`,用 STE clamp | `lat_truck.py:281` |
| `if a > b: branch1 else branch2` | `smooth_step(a, b, temp) · branch1 + (1 − smooth_step) · branch2` | `common.smooth_step` |
| `rate_limit(x_prev, x_new, max_rate·dt)` | STE clamp(允许梯度穿透速率限幅) | `common.rate_limit(... differentiable=True)` |
| `argmin`(轨迹最近点) | `.detach()` 隔离索引,仅在 y 值上回传 | `model/trajectory.py` |
| `lookup1d(table, x)` | 线性插值,PyTorch 原生可微 | `common.lookup1d` |

### 5.2 梯度爆炸的教训

做 smooth 近似时 **temperature 不能太小**。`sign(x)·min(|x|, L)` 如果用 `smooth_sign(x, temp=0.01)` 再乘 `smooth_lower_bound(|x|, L)`,链式求导会得到 `d/dx ≈ 1/temp = 100`,在 BPTT 150 步的链上直接爆炸到 10³⁰⁰。

教训:**先化简数学表达式再选近似方式**。`sign(x)·min(|x|, L) ≡ clamp(x, -L, L)`,用 STE clamp(导数恒=1),不需要 smooth_sign。

详见 `docs/bptt_gradient_explosion_analysis.md`。

### 5.3 TBPTT 截断

50Hz × 10 秒 = 500 步的计算图太深。做 **Truncated BPTT**:每 `tbptt_k` 步(默认 150,即 3 秒),把 plant 状态和控制器内部状态(`steer_*_prev`, `iir_acc.state`)**原地 detach**。
- 梯度不会回传超过 3 秒
- 但状态数值连续(仿真物理没断)
- 实测 k=150 和 full BPTT 收敛效果等价,但速度快 3-5 倍

## 6. 可微调参流程

### 6.1 Loss 构造

```python
loss = w_lat × lat_mse          # 横向误差,w=10
     + w_head × head_mse        # 航向误差,w=8
     + w_speed × speed_mse      # 速度误差,w=3
     + w_steer_rate × Δsteer²   # 方向盘变化率,w=0.05
     + w_acc_rate × Δacc²       # 加速度变化率,w=0.01
     + λ × Σ(p - p_init)²       # 参数 L2 正则,λ=0.01
```

### 6.2 Per-trajectory loss 归一化

直接把 48 条轨迹 loss 求和会让长轨迹 / 高速段主导梯度。所以 **每条轨迹的 loss 除以它在 epoch-1 时的 baseline**(先跑一遍初始参数下的 loss,冻结作为分母)。这样每条轨迹对梯度的贡献变成"相对改善率",公平性大幅提升。

### 6.3 参数投影

一些参数有物理约束(PID 增益必须 ≥ 0,时间参数不能负)。Adam 更新后,做一次 hard clamp:

```python
with torch.no_grad():
    for p in [T2_y, T3_y, T4_y, T6_y]:
        p.clamp_(min=0)
    switch_speed.clamp_(min=0.5, max=10.0)
```

### 6.4 批量并行训练(`train_batch.py`)

scalar 版本每 epoch 把 48 条轨迹串行跑,~40 分钟。**batched 版本** 把它们**同步推进**:每时间步所有 batch 元素一起过查表/RK4/MLP,TBPTT 一起 detach。

- 变长用 `padding + valid_mask`,padding 位置不算 loss
- PID/IIR 内部状态 shape `[B]` 独立演化
- 所有 `if/elif` 用 `torch.where` 精确复刻
- 单 epoch ~5 分钟(8.5× 加速)
- 训练 + 验证端到端走 batched,总链路 ~11 分钟

## 7. 被控对象的选择与 MLP 残差

| plant | 机理 | 是否带 MLP | 典型用途 |
|---|---|---|---|
| kinematic | 自行车模型 | 否 | 默认,调参快 |
| dynamic | 6-DOF 单轨 | 否 | 乘用车动力学基线 |
| hybrid_dynamic | 6-DOF + MLP | 是(残差) | 贴近 CarSim 真值 |
| **truck_trailer** | **牵引车+挂车 12D + RK4** | **可选 MLP** | **重卡专用(本项目主线)** |

**关键耦合**:MLP 残差模型是在**某组特定 base 参数**下训练的(相当于学"那组 base + Euler/RK4 和 CarSim 真值之间的差")。一旦你改 base 参数(比如 Cf/Cr 侧偏刚度,或 steering_ratio),MLP 看到的 state/control 分布就不再是训练时的,残差预测大概率错掉。所以物理参数和 MLP checkpoint 要锁死在一起用,改一个必须重训另一个。

这一点在本项目实际踩过坑 —— 把 truck plant 的侧偏刚度从 80000(上游训练值)改成 264000/335000(xlsx 实车推算值)后,MLP(_2) 的平均 lat RMSE 从 1.46 m 恶化到 3.46 m。详见 `docs/truck_vehicle_parameters.md`。

---

# Part II — AI 协作开发实录

## 1. 工作分工

这个项目**几乎完全**是靠 AI(Claude Code)写的。人(项目owner)的角色是**指挥 + 审稿 + 决策**:

| 角色 | 做什么 | 不做什么 |
|---|---|---|
| 人 | 提需求、定目标、判断方向、读代码 review、做架构决策、提供领域知识(车辆动力学、C++ 原码的行为) | 不亲手写大部分代码 |
| AI(Claude) | 读代码、做对比分析、写实现、跑测试、跑训练、画图、写文档 | 不做架构决策、不做方向判断 |

重点是:**人不当秘书,AI 不当决策者**。

### 典型一次交互

**人**:"看一下 sim 里纵向控制器的站位保护阈值。spec_v2 说是 1.0 kph,现在代码里是多少?如果不对请改过来,train_batch 里的 batched 版也要改。"

**AI**:
1. `Grep` 定位 `lon.py` 里 `if speed_kph_val > 10` — 两处(scalar 和 differentiable 路径)
2. `Grep` 定位 `train_batch.py` 里 `high_speed = speed_kph > 10.0` — 一处
3. 分别 Edit,把 10 → 1.0
4. 跑 `pytest tests/ -q` 验证没打破测试
5. 汇报改动范围 + 测试结果

这个粒度的任务,人出一句话,AI 做完要报告的是"具体动了哪几个文件、第几行、测试是不是绿",人只需扫一眼就能定性。

## 2. 让 AI 能干活的前置条件

### 2.1 CLAUDE.md —— 项目的"门面"

仓库根有 `CLAUDE.md`,`sim/` 子目录还有 `sim/CLAUDE.md`。每次 AI 启动都会自动读这些。里面写清:

- **目录结构 + 每个模块的作用**
- **常用命令**(跑测试、跑 demo、训练)
- **代码风格约定**(注释最小化、commit 规范)
- **沟通风格**(用自然语言解释机制,少堆代码黑话)
- **已有设计决策的原因**(为什么选某技术路线)

CLAUDE.md **不是 README**。README 给用户看,CLAUDE.md 给 AI 看。AI 看了这个文件应该能 5 分钟内进入状态,知道"这项目在干什么、哪里可以改、哪里不能动"。

### 2.2 自动保存的"记忆"(MEMORY.md)

Claude Code 会在 `.claude/projects/<project>/memory/` 下维护一套 markdown:

- `user_role.md` — 人是谁、擅长什么
- `feedback_*.md` — 人给过的具体反馈(比如"解释机制时用自然语言,不要堆代码黑话")
- `project_*.md` — 项目当前在做什么、已知问题、里程碑
- `reference_*.md` — 外部资源(比如"C++ 清稿在 src/mlp/control/,不是 output/")

AI 每次开新对话,默认加载 `MEMORY.md` 索引,按需拉取某个具体文件。这解决了 "每次都要解释一遍上下文" 的问题。

### 2.3 spec 文档 = 真正的契约

`docs/controller_spec_v2.md` 是控制器算法的权威规格书,逐步描述每个计算公式 + 参数依赖 + 查找表。这份文档承担了两个角色:

- **AI 理解代码时的对照物**(代码和 spec 不一致就是 bug 或 spec 过时)
- **人验证 AI 改动正确性的标尺**

实操中,spec 也会被 AI 修正(例如这次 OCR 错的 `kMin_prev_dist=4.0` 就是通过截图复核后改回 5.0)。但 spec 始终是 source of truth,代码向它看齐。

## 3. 问题排查的套路

### 3.1 现象 → 差异 → 根因

不要一上来就改代码。让 AI **先定位现象在代码里的哪个位置、和什么"应该的行为"有差异、差异的根因是什么**。典型对话:

**人**:"lat RMSE 在 55 kph 换道场景上 5 米,很离谱,查一下原因。"

**AI 应做**:
1. 把仿真轨迹画出来(matplotlib),定位什么时候误差起飞
2. 同时画参考轨迹、实际轨迹、方向盘角、车轮转角
3. 看是"方向盘输出大但车轮转角小"(转向延迟?)还是"输入输出都大但 plant 慢"(动力学太硬?)
4. 再根据现象假设根因,再动代码

**AI 不应做**:一看到 lat RMSE 大就乱调控制器参数或改仿真。

### 3.2 保留中间产物

每次训练或验证都留下可复现证据:

- **训练日志**:每 epoch 打印 loss 分项、梯度范数、参数快照、NaN 数量,写入 `results/training/<plant>/<timestamp>/loss_curve.png` + `experiment_log.yaml`
- **V1 对比图**:baseline vs tuned 的 49 场景轨迹 + 横向误差 + 方向盘角 + 加速度四组图,放在 `results/validation/`
- **参数投影记录**:哪些参数被 clamp 到了边界

AI 排查时不是靠 "我记得上次是这样",而是去读 `experiment_log.yaml`、读 loss_curve,证据驱动。

### 3.3 有限差分:梯度验证的核武器

一次"梯度爆炸"问题里,loss 正常收敛但梯度范数飙到 10³⁰⁰。初步怀疑是 smooth 近似的 temperature 太小。怎么验证?

```python
# 用有限差分算"真实"梯度
def numerical_grad(param, eps=1e-4):
    orig = param.item()
    param.fill_(orig + eps); loss_plus = run()
    param.fill_(orig - eps); loss_minus = run()
    return (loss_plus - loss_minus) / (2 * eps)

# 对比 autograd 的梯度
print("autograd:", param.grad.item())
print("numerical:", numerical_grad(param))
```

真实梯度 O(1),autograd 给的 10³⁰⁰ —— 确认是 smooth 近似的反向导数计算路径有问题,和"loss 曲面"没关系。这个诊断让修复方向明确(改用 STE clamp),而不是去动超参。

## 4. 可观测性:为 AI 设计的代码

这条最容易被忽略。"可观测性" 在传统软件里是给运维看日志,在 AI 协作里是 **让 AI 第二天能独立读懂昨天发生了什么**。

### 4.1 训练脚本要"话多"

以前的训练脚本可能是这样:
```python
for epoch in range(n_epochs):
    loss = train_one_epoch()
    print(f"Epoch {epoch}: loss={loss}")
```

AI 看到这份日志能干什么?什么都不能。改成这样:

```python
print(f"Epoch {epoch:2d}/{n_epochs} | "
      f"loss={loss:.4f} ({pct:+.2f}%) | "
      f"lat_rmse={lat_rmse:.4f}m head_rmse={head_rmse:.4f}rad speed_rmse={speed_rmse:.4f}m/s | "
      f"grad_norm={grad_norm:.2f} (clipped {n_clip}/{n_total}) | "
      f"nan_grads={nan_count} | "
      f"elapsed={dt:.1f}s")
```

加上每 N epoch 打参数快照:

```
=== 参数快照 @ epoch 6 ===
T2_y:       [1.50 1.50 1.50 1.50 1.50 1.50 1.50]   (不变)
T4_y:       [0.00 0.00 0.12 0.18 0.22 0.25 0.25]   (初值 0.3,平均 -33%)
station_kp: 0.172 (初 0.25, -31%)
```

这样 AI 第二天读日志,不用跑训练就能知道 T4 是主要收敛方向、station_kp 在变小。

### 4.2 验证脚本要出图、出 yaml

`optim/post_training.py` 会:
- 自动生成 49 场景的 4 组对比图(轨迹/横向误差/方向盘/加速度)
- 把每个场景的 lat_rmse/head_rmse/speed_rmse 写进 `experiment_log.yaml`

图是给人看的,yaml 是给 AI 读的。**两个都要**。yaml 让 AI 可以 `yaml.safe_load` 后直接排序、聚合、求均值,比解析终端日志鲁棒得多。

### 4.3 commit 就是时光机

本项目严格遵守:

- 每次动代码 + 核心文档 + 结果图,**一个 commit 一起上**
- commit message 中文,前缀 `[sim] / [docs] / [extract]`
- 一个完整任务的多个步骤合并成一个 commit,不拆碎片
- commit 后立即 push 到远端

AI 在"理解项目演进"时,`git log --oneline` 是第一入口。如果 commit 都是"wip"、"fix",AI 读不出故事。如果每个 commit 都描述"做了什么、为什么做",AI 就能建立起这个项目的演化时间线。

### 4.4 把 "AI 调试会用到的脚本" 写成独立工具

例子:`sim/health_check.py` — 一键体检,跑测试 + 跑基线 demo + 梯度健康检查。AI 排查问题时先跑一遍这个,能把"环境问题/代码问题/数据问题"三类快速区分开。

类似的:`optim/validate_batch.py` 支持 `--config-a / --config-b / --disable-mlp-a` 做自定义 A/B 对比 —— 任何"我想知道 X vs Y 哪个效果好"的问题,一行命令能回答。

## 5. 指挥 AI 的几条经验

### 5.1 模糊的需求要自己先想清楚

反例:"优化一下控制器"(方向是啥?收敛到什么?)
正例:"lat RMSE 在 clothoid_55kph 上 2.5m,想降到 1m 以内。先读当前参数 + 跑一次 baseline,再提 3 种可能方向我挑。"

### 5.2 小步小改,每步可验证

AI 连续改 10 个文件 +200 行后,"一起测试",基本等于开盲盒。更好的节奏:改 1 个文件 → `pytest tests/` → 跑 demo 确认没坏 → 提交 → 改下一个。

### 5.3 给 AI "降智" 的心理准备

Claude 会犯的典型错误:

- **幻想**:编造一个不存在的函数名(防范:让它 Grep 验证)
- **过度工程**:给 `a+b` 加 try/except(防范:CLAUDE.md 里明确 "don't add error handling for scenarios that can't happen")
- **对称思维**:改了 scalar 版忘了改 batched 版(防范:显式提醒"记得同步 train_batch")
- **旧记忆覆盖新状态**:memory 里说 T5 无梯度,其实早就修好了(防范:memory 同步维护)

这些用"人读一遍 diff"就能抓住。review 是必须的,不是可选。

### 5.4 小问题先自己查,大问题才上 AI

查"某变量叫啥、某文件在哪"这种:你直接 grep 比等 AI 回来更快。AI 适合的是"给我一张表,对比 sim 参数和 xlsx 参数,再按公式推算侧偏刚度" —— 那种跨文件 + 跨领域 + 需要组织结论的任务。

## 6. 本项目具体用到的 AI 工具栈

| 工具 | 用途 |
|---|---|
| **Claude Code** (CLI) | 主力。Read/Edit/Write/Bash/Glob/Grep/WebSearch,pre-alpha 时期就开始用 |
| **Agent / Subagent** | 大查询并行化(比如"给我扫一下所有 sim/ 下用到 corner_stiff 的地方")不污染主上下文 |
| **TaskCreate/TaskUpdate** | 多步任务的进度跟踪(这次对话本身就在用) |
| **WebSearch/WebFetch** | 查轮胎经验系数、查 SAE paper、查 Pacejka 参数 |
| **git** | 版本时光机,AI 自己会用 `git log/diff/blame` 追历史 |

---

## 7. 回看:这个项目里 AI 实际贡献了什么

按代码行数估,AI 写的占 95%+。按**关键决策**,AI 贡献约等于零 —— 架构、技术选型、调参方向全部是人拍板。

但有几个细节是 AI 出人意料地帮上忙的:

1. **找出 spec 和 C++ 源码的不一致**:对照 1400 行 spec 和 3000 行 C++,人工做要 3 天,AI 半小时搞定
2. **做单位 / 坐标对齐的推导**:"这里方向盘角是 deg,plant 接口要 rad,除一下 steer_ratio,再 × DEG2RAD" —— AI 在写代码时不会忘
3. **写文档**:这份文档就是 AI 写的。人改了 3 处措辞
4. **跑实验**:训练 + 验证 + 画图 + 对比,命令行拼接、yaml 解析、matplotlib 调教,这些琐事全部交给 AI

失败的情况也有:

1. 第一次改 TBPTT 的时候,AI 想"优化"一下,偷偷改了默认 k 值 —— 被 review 发现回退
2. 让 AI 写 loss 归一化时,它第一版写错了(用了平均 loss 当分母,应该用 per-trajectory baseline)—— 第二轮 prompt 指明期望后改对

## 8. 结语

用 AI 做项目**不是把写代码的脑力劳动外包**。人仍然要想清楚系统长什么样、哪里需要抽象、目前的瓶颈是什么。AI 是**一个打字飞快、知识面宽、但没有目标感的同事** —— 你给它一个清晰的 ticket,它能按时交付;你给它一个模糊的需求,它会给你 10 个看起来都对但没法用的东西。

所以这个项目能跑起来的根本原因不是 Claude 多聪明,而是:
1. **CLAUDE.md + MEMORY 让 AI 有了长期记忆**
2. **spec + 单元测试 + 对比图让 AI 的每步改动都能被快速验证**
3. **人保留了所有方向判断权,AI 只负责执行**

这套工作方式适用性广,任何"代码量中等 + 领域知识可传递 + 明确有 ground truth(spec/测试/数据)"的项目都能复制。差的反而是算法研究这种"方向不明、ground truth 要自己定义"的场景 —— 那种还是得人自己跑实验。
