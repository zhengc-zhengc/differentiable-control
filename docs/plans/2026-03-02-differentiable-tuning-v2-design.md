# V2 可微调参设计文档

## 目标

将 V1 numpy 控制器原地升级为 PyTorch 版本，支持端到端可微调参。优化横向（LatControllerTruck）和纵向（LonController）的全部可调参数，通过梯度下降最小化跟踪误差。每次调参结果保存为完整 YAML 文件（带 commit hash），可随时加载任意版本的控制器参数。

## 方案选择

**方案 B：原地升级** — 将 V1 代码从 numpy 直接转为 PyTorch，通过 `differentiable=True/False` 开关控制非光滑操作的实现方式。`False` 模式下行为与 V1 numpy 版完全一致。

选择理由：
- V1 设计时已预留 V2 转换路径（参数集中在 YAML，不硬编码）
- 一套代码 + 开关，维护成本最低
- `differentiable=False` 就是 V1 等效，原有测试直接复用验证

## 架构

```
sim/
├── common.py          # 升级：numpy→torch，添加 smooth_* 可微原语
├── config.py          # 升级：支持 nn.Parameter 注册 + YAML 保存（带 commit hash）
├── configs/
│   ├── default.yaml        # V1 默认参数（不动）
│   └── tuned/              # 新增：调参结果目录
│       └── tuned_{commit}_{timestamp}.yaml
├── trajectory.py      # 升级：torch 化，nearest query 用 detached argmin
├── vehicle.py         # 升级：torch 化
├── controller/
│   ├── lat_truck.py   # 升级：torch 化 + differentiable 开关
│   └── lon.py         # 升级：torch 化 + differentiable 开关
├── sim_loop.py        # 升级：torch 化，支持梯度回传
├── train.py           # 新增：可微调参训练入口
└── run_demo.py        # 不动（调参后用调参 YAML 跑可视化验证）
```

核心原则：所有模块统一使用 `torch.Tensor`，不再有 numpy。`differentiable` 开关只影响非光滑操作的实现方式，不影响数据类型。

## 可微原语库（common.py）

### lookup1d — 不需要特殊处理

优化目标是表的 y 值（`nn.Parameter`），x 轴（车速）不参与优化。PyTorch 线性插值对 y 值天然可微：`y = y_lo * (1-t) + y_hi * t`，`dy/d(y_lo) = (1-t)`，`dy/d(y_hi) = t`。

### smooth_clamp

```python
def smooth_clamp(x, lo, hi, temp=0.1):
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    return mid + half * torch.tanh((x - mid) / (half * temp))
```

- `differentiable=False`：`torch.clamp`（与 numpy clip 一致）
- `differentiable=True`：`smooth_clamp`（边界处仍有梯度）

### smooth_rate_limit

本质是对 delta 做 clamp，复用 smooth_clamp：

```python
def rate_limit(prev, curr, rate, dt, differentiable=False):
    delta = curr - prev
    max_delta = rate * dt
    if differentiable:
        return prev + smooth_clamp(delta, -max_delta, max_delta)
    else:
        return prev + torch.clamp(delta, -max_delta, max_delta)
```

### smooth_sign

```python
def smooth_sign(x, temp=0.01):
    return torch.tanh(x / temp)
```

### normalize_angle

```python
def normalize_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))
```

`atan2(sin, cos)` 处处可微（除 ±π），正常跟踪角度远离该区域。两种模式均可用此实现。

### PID 积分器

积分累加和比例运算天然可微。饱和限幅用 `smooth_clamp` / `torch.clamp` 按开关选择。

### 温度超参数

所有 smooth 函数的 temperature 集中管理，作为训练超参数（不参与优化）：

```yaml
smooth_temps:
  clamp: 0.1
  sign: 0.01
  switch: 1.0
```

## 控制器转换

### LatControllerTruck

| V1 操作 | differentiable=True | differentiable=False |
|---------|--------------------|--------------------|
| 8× lookup1d | torch 线性插值（天然可微） | 同左 |
| sign(error) | smooth_sign(e, temp) | torch.sign |
| min(max_theta, abs(e)) | smooth_clamp | torch.clamp |
| 3× rate_limit | smooth_rate_limit | hard rate_limit |
| clamp 输出 | smooth_clamp | torch.clamp |

可优化参数：8 张表 y 值 + kLh。rate_limit / min_dist 默认不优化（安全约束），留接口可选。

### LonController

| V1 操作 | differentiable=True |
|---------|---------------------|
| if speed > 10 | sigmoid 混合两分支 |
| if station <= 0.25 / >= 0.8 | 链式 sigmoid 混合 |
| if speed <= switch_speed | sigmoid 混合 PID 增益 |
| if curvature < -0.0075 | sigmoid 混合限制系数 |
| if abs(v) < 1.5 | sigmoid 混合 |
| 5× lookup1d | torch 线性插值 |
| 多处 clamp | smooth_clamp |

`differentiable=False` 时，所有条件用 `.item()` 取标量做 Python if/else，执行路径与 V1 完全一致：

```python
if differentiable:
    w = torch.sigmoid((speed - threshold) / temp)
    result = w * branch_a + (1 - w) * branch_b
else:
    if speed.item() > threshold:
        result = branch_a
    else:
        result = branch_b
```

可优化参数：3 组 PID kp/ki + switch_speed + 5 张表 y 值 + preview_window + iir_alpha + acc_use_preview_a。

## 轨迹最近点查询

hard argmin 不可微，但无需可微化。理由：优化的是控制器参数，不是轨迹。小参数扰动改变车辆位置，但"最近参考点是哪个"这一离散选择几乎不变。

策略：**detached argmin** — `torch.no_grad()` 下找最近点索引，后续误差计算正常回传梯度。梯度通过 `(vehicle_pos - ref_point)` 流回车辆状态。

```python
def query_nearest_by_position(self, x, y):
    with torch.no_grad():
        dists = (pts_x - x)**2 + (pts_y - y)**2
        idx = torch.argmin(dists)
    return self.points[idx]
```

## 训练 Pipeline

### 参数注册

`DiffControllerParams(nn.Module)` 把所有可优化参数注册为 `nn.Parameter`。控制器每步从该 module 读参数。

### Loss 函数

```python
L = (w_lat * mean(lateral_error²)
   + w_head * mean(heading_error²)
   + w_speed * mean(speed_error²)
   + w_steer * mean(steer_rate²)
   + w_acc * mean(acc_rate²))
```

权重 `w_*` 为训练超参数。

### 训练循环

```python
for epoch in range(n_epochs):
    for trajectory in trajectories:
        history = sim_loop(trajectory, params, differentiable=True)
        loss = tracking_loss(history)
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

sim_loop 展开整个仿真（N 步 × 50Hz），loss.backward() 通过时间反向传播。

### 优化器

Adam，支持分组学习率：
- 查找表参数：lr=5e-4（变化应平滑）
- PID 增益：lr=1e-3

## 参数 YAML 保存

### 文件命名

```
sim/configs/tuned/tuned_{commit_hash_8}_{YYYYMMDD_HHMMSS}.yaml
```

commit hash 通过 `git rev-parse --short HEAD` 获取。

### 文件内容

与 `default.yaml` 完全相同的结构，加元信息头：

```yaml
meta:
  base_config: default.yaml
  commit: ad5db1b3
  timestamp: "2026-03-02T15:30:00"
  final_loss: 0.0234
  epochs: 100
  trajectories: [circle, sine, combined]

vehicle:
  wheelbase: 3.5
  # ...
lat_controller:
  tables:
    T1: { speeds: [...], values: [...] }
    # ...
lon_controller:
  station_pid: { kp: 0.31, ki: 0.0 }
  # ...
```

任何 tuned YAML 可直接被 config.py 加载，传给 run_demo.py 可视化验证：

```bash
python sim/run_demo.py --config sim/configs/tuned/tuned_ad5db1b3_20260302_153000.yaml
```

## differentiable=False 一致性保证

核心策略：
1. 全部用 `torch.Tensor`，不混用 numpy
2. `False` 模式下非光滑操作用 torch 等价硬函数（`torch.clamp`、`torch.sign`）
3. 条件分支用 `.item()` 取标量做 Python if/else，执行路径与 V1 完全一致
4. V1 的 43 项测试全部在 `False` 模式下通过（允许浮点精度差异 < 1e-6）
