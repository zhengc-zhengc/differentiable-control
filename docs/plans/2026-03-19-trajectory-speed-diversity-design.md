# 轨迹多样性 + 速度波动 + CLI 简化设计

> 日期：2026-03-19

## 背景

当前训练默认仅 6 条轨迹覆盖 3 个速度段（18/35/55 kph），且所有轨迹参考速度为常量。
问题：
1. 缺失 5/25/45 kph 三个速度段，查找表 T1-T8 在这些断点附近的插值未被训练
2. 参考速度恒定 → 纵向控制器仅做"维持恒速"，增减速能力未被激发
3. CLI 需要逐条输入轨迹名，48+ 条轨迹时不可用

## 设计目标

- 全 6 个速度段覆盖（5/18/25/35/45/55 kph）
- 每个速度段统一的轨迹类型集
- 引入变速轨迹：弯前减速/弯后加速 + 梯形加减速
- CLI 简化：输入类型名 → 自动展开全速度段

## 标准类型集

### 恒速轨迹（5 种）

| 类型名 | 说明 | 几何特征 |
|--------|------|---------|
| `lane_change` | 单换道 | cosine 曲线，C2 连续 |
| `double_lc` | 双换道 | 左换道 → 直线 → 右换道 |
| `clothoid_left` | 左转 clothoid | 直线 → clothoid → 圆弧 → clothoid → 直线 |
| `clothoid_right` | 右转 clothoid | 同上，右转 |
| `s_curve` | S 弯 | 左弧 → 右弧，出口航向 ≈ 入口 |

### 变速轨迹（3 种）

| 类型名 | 说明 | 速度模式 |
|--------|------|---------|
| `combined_decel` | 直线+弯道+直线 | 弯前减速 → 弯道低速 → 弯后加速 |
| `clothoid_decel` | clothoid 弯道 | 同上 |
| `lc_accel` | 单换道 | 梯形加减速（±15% 巡航速度）|

### 特殊轨迹（仅验证）

| 类型名 | 说明 |
|--------|------|
| `park_route` | 园区综合路线（内置变速+停靠，commit 06a1912）|

**合计**：8 类型 × 6 速度段 = 48 条训练轨迹；验证额外 +1 park_route = 49 条。

## 速度段参数表

几何参数随速度缩放，保证每个速度段的轨迹都符合物理合理性。

| 速度 (kph) | speed (m/s) | lc_len (m) | r_clothoid (m) | clothoid_angle | r_combined (m) | r_scurve (m) | scurve_angle |
|-----------|-------------|-----------|----------------|---------------|----------------|-------------|-------------|
| 5 | 1.4 | 30 | 20 | π/2 | 15 | 30 | π/4 |
| 18 | 5.0 | 50 | 40 | π/2 | 30 | 50 | π/4 |
| 25 | 6.9 | 40 | 45 | π/2 | 35 | 50 | π/4 |
| 35 | 9.7 | 55 | 50 | π/2 | 40 | 60 | π/4 |
| 45 | 12.5 | 75 | 60 | π/2 | 50 | 70 | π/4 |
| 55 | 15.3 | 90 | 70 | π/3 | 60 | 80 | π/4 |

## 变速剖面算法

### A) 弯前减速/弯后加速（`combined_decel`, `clothoid_decel`）

基于曲率约束的速度规划：

```
输入：轨迹点序列 pts, 巡航速度 v_cruise
参数：a_lat_max = 2.0 m/s², decel_rate = 0.8 m/s², accel_rate = 0.5 m/s²

1. 曲率速度上限：v_max[i] = min(v_cruise, sqrt(a_lat_max / max(|κ[i]|, ε)))
2. 前向约束（减速）：v[i] = min(v_max[i], sqrt(v[i-1]² + 2·decel_rate·ds))
3. 后向约束（加速）：v[i] = min(v[i], sqrt(v[i+1]² + 2·accel_rate·ds))
4. 更新 pts[i].v, 计算 pts[i].a = (v[i] - v[i-1]) / dt
```

### B) 梯形加减速（`lc_accel`）

```
输入：轨迹点序列 pts, 基础速度 v_base
参数：delta_ratio = 0.15 (±15%), accel_rate = 0.5 m/s²

v_lo = v_base × (1 - delta_ratio)
v_hi = v_base × (1 + delta_ratio)

速度剖面：v_lo → 加速到 v_hi → 巡航 → 减速到 v_lo
各阶段等分总路程，加减速段长度由 accel_rate 决定
```

## CLI 设计

### 训练

```bash
# 指定类型 → 展开到全 6 速度段
python optim/train.py --trajectories lane_change clothoid_decel lc_accel

# 默认 → 全类型×全速度段 (48 条)
python optim/train.py
```

### 验证

```bash
# 指定类型 → 展开到全速度段 + 自动追加 park_route
python optim/post_training.py --config xxx.yaml --trajectories lane_change clothoid_decel

# 默认 → 全量 (49 条)
python optim/post_training.py --config xxx.yaml
```

## 代码改动

### trajectory.py

- 新增 `_SPEED_BANDS_KPH = [5, 18, 25, 35, 45, 55]`
- 新增 `_SPEED_PARAMS` 速度参数表
- 新增 `_TRAJECTORY_TYPES` 类型注册表（8 种）
- 新增 `expand_trajectories(type_names=None)` 展开函数
- 新增 `apply_curvature_speed_profile(pts, v_cruise, ...)`
- 新增 `apply_trapezoidal_speed_profile(pts, v_base, ...)`

### train.py

- `--trajectories` 参数语义变更：接收类型名（非全名）
- 默认值从 6 条硬编码改为 `expand_trajectories()`（全量 48 条）
- 移除旧的 `_TRAJECTORY_BUILDERS` 扁平字典

### post_training.py

- `_EVAL_SCENARIOS` 改用 `expand_trajectories()` + park_route
- `--scenarios` 参数改为 `--trajectories`，语义同训练
- 验证默认全量 49 条

## 向后兼容

- 旧的 `_TRAJECTORY_BUILDERS` 和 `_EVAL_SCENARIOS` 在迁移完成后删除
- `run_demo.py` 等其他脚本如直接调用 trajectory 生成函数则不受影响
