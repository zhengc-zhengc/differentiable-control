# 控制器算法实现规格书

目的：根据此文档可在 Python 等任意语言中完整复现横纵向控制器，复现结果与 C++ 原始代码对物理信号的处理效果等价。

执行频率：50 Hz，步长 dt = 0.02 s

车型选择：use_truck_lat_control = true → 重卡横向控制器；false → 电动牵引车横向控制器。纵向控制器两者共用。

## 0. 符号约定

| 符号 | 含义 |
|------|------|
| clamp(x, lo, hi) | max(lo, min(hi, x)) |
| normalize_angle(a) | 将弧度归一化到 (-π, π] |
| sign(x) | +1/0/-1，|x| < 1e-3 时取 0 |
| deg2rad | π / 180 |
| rad2deg | 180 / π |
| [prev] | 上一控制周期的值（内部状态） |

## 0.1 系统信号流总览

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │              ControlComponent（定时组件，50Hz 主循环）                 │
 │                                                                      │
 │  GlobalPose ─────┬──> 横向控制器 ──────────┬──> 方向盘转角指令        │
 │  （位置/航向）     │   （重卡版 or 电拖版）   │    steering_target (°)  │
 │                  │                         │                         │
 │  CCANInfoStruct ─┤                         │    远预瞄曲率 ──────┐    │
 │  （车速/横摆率等） │                         │    curvature_far   │    │
 │                  │                         │                    │    │
 │  ADCTrajectory ──┤                         │                    │    │
 │  （规划轨迹）      │                         │                    │    │
 │                  └──> 纵向控制器  <──────────┘                    │    │
 │                       LonController                              │    │
 │                       │  读取 curvature_far（横→纵唯一耦合信号）    │    │
 │                       │                                          │    │
 │                       ├──> 驱动扭矩 target_torque (Nm，加速时)    │    │
 │                       ├──> 制动减速度 brake (m/s²，减速时)        │    │
 │                       └──> 档位请求 gear_location (N/D/P)        │    │
 └──────────────────────────────────────────────────────────────────────┘
```

**执行顺序**：横向控制器先执行，将远预瞄曲率 `path_curvature_far` 写入 ControlCommand；
纵向控制器后执行，读取该曲率用于急弯减速判断。信息流**单向**（横向→纵向），
两个控制器之间无反向依赖。

## 1. 公共基础组件

### 1.1 Lookup1D — 分段线性插值

```python
def lookup1d(table: list[tuple[float, float]], x: float) -> float:
    """table 为 (index, value) 有序列表，x 在范围外时夹紧到边界值。"""
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
```

### 1.2 RateLimitFilter — 速率限制

```python
def rate_limit(prev: float, target: float, rate: float, dt: float) -> float:
    """rate 单位与 target/dt 相同（如 °/s 或 m/s²/step）。"""
    max_delta = rate * dt
    return prev + clamp(target - prev, -max_delta, max_delta)
```

### 1.3 IIR 低通滤波器（一阶）

代码中使用 {num=[0,0,1], den=[0,α,1]} 的传递函数，等价于：

```python
class IIR:
    def __init__(self, alpha: float):
        self.alpha = alpha  # 时间常数 τ（秒），实际增益 ≈ dt/(τ+dt)
        self.y_prev = 0.0
    def update(self, x: float) -> float:
        # 等效：y = x - alpha * y_prev  (den 格式的递推式)
        y = x - self.alpha * self.y_prev
        self.y_prev = y
        return y
    def reset(self):
        self.y_prev = 0.0
```

代码中的 alpha 值：纵向 acc_out 滤波 α=0.15；电动牵引车横向 LPF1 α=0.01，LPF2 α=0.03。

### 1.4 带阻（陷波）滤波器（仅电动牵引车横向）

```python
class BandstopFilter:
    # 传递函数系数（二阶 IIR）
    # num = [1.0, 52.51, 2527], den = [1.0, 100.54, 2527]
    # H(z) = (num[0]*z^2 + num[1]*z + num[2]) / (den[0]*z^2 + den[1]*z + den[2])
    # 离散递推（标准 Direct Form II Transposed）：
    def __init__(self):
        self.num = [1.0, 52.51, 2527.0]
        self.den = [1.0, 100.54, 2527.0]
        self.w = [0.0, 0.0]  # 状态
    def update(self, x: float) -> float:
        w0 = x - self.den[1]*self.w[0] - self.den[2]*self.w[1]
        y  = self.num[0]*w0 + self.num[1]*self.w[0] + self.num[2]*self.w[1]
        self.w[1] = self.w[0]
        self.w[0] = w0
        return y
```

### 1.5 PID 控制器（含抗积分饱和）

```python
class PID:
    def __init__(self, kp, ki, kd, integrator_enable, integrator_saturation):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_enable = integrator_enable
        self.sat = integrator_saturation
        self.integral = 0.0
        self.prev_error = 0.0
    def control(self, error: float, dt: float) -> float:
        if self.integrator_enable:
            self.integral = clamp(self.integral + error * dt, -self.sat, self.sat)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
    def set_pid(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
```

### 1.6 TrajectoryAnalyzer — 轨迹查询接口

轨迹为一组时序路径点，每个点包含：

| 字段 | 单位 | 含义 |
|------|------|------|
| x, y | m | ENU 坐标 |
| theta | rad | 切线方向（航向角） |
| kappa | 1/m | 路径曲率（左转为正） |
| v | m/s | 参考速度 |
| a | m/s² | 参考加速度 |
| s | m | 累积弧长 |
| relative_time | s | 相对轨迹起点的时间 |
| absolute_time | s | 绝对时间戳 |

```python
def query_nearest_by_position(traj, x, y):
    """返回距离 (x,y) 欧氏距离最近的轨迹点。"""

def query_nearest_by_relative_time(traj, t_rel):
    """按 relative_time 线性插值，超界则夹紧到端点。"""

def query_nearest_by_absolute_time(traj, t_abs):
    """按 absolute_time 线性插值，超界则夹紧到端点。"""

def to_frenet(traj, x, y, theta_rad, v_mps, matched_point):
    """Frenet 坐标变换。返回 (s_matched, s_dot, d, d_dot)。
    heading_err = normalize_angle(theta_rad - matched_point.theta)
    s_dot = v * cos(heading_err)
    d       = cos(matched_point.theta)*(y - matched_point.y)
              - sin(matched_point.theta)*(x - matched_point.x)
    d_dot = v * sin(heading_err)
    s_matched = matched_point.s + projection_along_path
    """
```

## 2. 横向控制器 A：重卡版（LatControllerTruck）

### 2.1 接口

**输入**

| 变量名 | 来源 | 单位 | 说明 |
|--------|------|------|------|
| x, y | GlobalPose.position_enu | m | 车辆 ENU 坐标 |
| yaw_deg | GlobalPose.euler_angles.z | 度→rad | 车辆航向角，需转弧度 |
| speed_kph | CCANInfoStruct.ccan221_abs_vehspdlgt | km/h | 车速 |
| yawrate | CCANInfoStruct.ccan242_esp_yawrate | rad/s | 横摆角速度 |
| steer_feedback | CCANInfoStruct.ccan175_eps1_eps1_steeringwheelangle | 度 | 方向盘实际角度（仅用于失能时跟随） |
| trajectory | ADCTrajectory | — | 规划输出点列表 |
| ctrl_enable | bool | — | 控制使能 |

**输出**

| 变量名 | 写入字段 | 单位 |
|--------|---------|------|
| steering_target | cmd.steering_target | 度 |
| curvature_current | cmd.path_curvature_current | 1/m |
| curvature_near | cmd.path_curvature_near | 1/m |
| curvature_far | cmd.path_curvature_far | 1/m（纵向控制器读取） |

### 2.2 信号流

```
  ┌──── 查表层（全部以 speed_kph 车速为索引）─────────────────────────────┐
  │  T1→最大误差角   T2→预瞄距离时间   T3→收敛时间   T4→阻尼时间         │
  │  T5→近预瞄时间   T6→远预瞄时间     T7→最大转角   T8→侧滑修正         │
  └──────────────────────────────────────────────────────────────────────┘
                              │ (Step 1: 查表)
                              ▼
  车辆位置 x, y ────────> ┌──────────────┐
  规划轨迹 trajectory ──> │  轨迹查询     │──> currt 当前匹配点（位置最近）
                         │  (Step 2)     │    near  近预瞄点（+0.1s）
                         └──┬───────────┘    far   远预瞄点（+1.0s）
                            │
  车辆航向 yaw_rad ────>┌───▼───────────┐
  轨迹航向 currt.θ ───> │  误差计算      │──> lateral_error  横向偏差(m,左正)
                        │  (Step 3)      │    heading_error 航向偏差(rad)
                        └───┬───────────┘    curvature_far  远点曲率(1/m)
                            │
  横摆角速度 yawrate ─> ┌───▼───────────┐
  车速 speed_mps ─────> │  实际航向角     │──> real_theta 实际航向误差
                        │  (Step 4)      │   = -heading_error
                        └───┬───────────┘   （kLh=0 铰接修正退化为恒等）
                            │
  横摆角速度 yawrate ─> ┌───▼───────────┐
  远点曲率 κ_far ─────> │  航向角变化率   │──> real_dt_theta 航向误差变化率
  车速 speed_mps ─────> │  (Step 5)      │   = -(ω - κ_far × v)
                        └───┬───────────┘
                            │
  T1 最大误差角 ──────> ┌───▼───────────┐
  T2 预瞄距离 ────────> │  目标航向角     │──> target_theta 期望航向（限幅后）
  横向偏差 lat_err ───> │  (Step 6)      │    target_dt_theta 期望变化率
                        └───┬───────────┘
                            │
  T3 收敛时间 ────────> ┌───▼───────────┐
  T4 阻尼时间 ────────> │  目标曲率      │──> target_curvature 反馈曲率
                        │  (Step 7)      │  = -(Δθ + Δθ̇×T_dt) / 收敛距离
                        └──┬─────┬──────┘
                           │     │
  T8 侧滑修正 ──────>┌────▼───┐ ┌▼──────────┐
  轴距 wheelbase ───> │反馈转角 │ │ 前馈转角   │<── 远点曲率 κ_far
  转向比 steer_ratio >│(Step 8)│ │ (Step 9)  │    （道路曲率直接补偿）
                      └──┬────┘ └──┬────────┘
              速率限制 120°/s   速率限制 165°/s
                         │        │
  T7 最大转角 ─────>┌───▼────────▼───┐
                   │  合并 + 限幅    │──> steering_target 方向盘转角(°)
                   │  (Step 10)      │    总速率限制 300°/s
                   └─────────────────┘
```

**信号流小结**：
- **反馈路径**（Steps 3–8）：车辆偏离轨迹的位置/航向误差 → 目标曲率 → 反馈转角。T2（预瞄距离）控制比例增益，T1（最大误差角）限制输出幅度。
- **前馈路径**（Step 9）：远预瞄点的道路曲率 `κ_far` 直接转换为前馈转角，用于补偿弯道——即使车辆没有偏差，弯道中也需要转向。
- **合并**（Step 10）：反馈 + 前馈 → 幅值限制（T7）→ 总速率限制（防止方向盘跳变）→ 输出。
- **无积分器**：全部是比例 + 前馈结构，不累积历史误差。优点是无超调风险，缺点是可能有稳态误差。

### 2.3 车辆固定参数（不调整）

| 参数 | 典型值 | 说明 |
|------|--------|------|
| wheelbase | vehicle_param.wheel_base (m) | 轴距 |
| steer_ratio | vehicle_param.steer_ratio | 转向传动比（方向盘角/车轮转角） |
| min_speed_prot | 0.1 km/h | 速度保护下限 |

**硬编码常量（代码级，不在配置文件）**

| 常量 | 值 | 含义 |
|------|------|------|
| kLh | 0.0 m | 铰接轴距（保留接口，当前为 0） |
| kRate_limit_fb | 120 °/s | 反馈通道速率限制 |
| kRate_limit_ff | 165 °/s | 前馈通道速率限制 |
| kRate_limit_total | 300 °/s | 合并后总速率限制 |
| kMin_prev_dist | 5.0 m | 预瞄距离最小值 |
| kMin_reach_dis | 3.0 m | 收敛距离最小值 |

### 2.3 可调控制器参数（查找表）

重要说明：以下所有查找表在代码中均以**当前车速（km/h）**作为查询索引，与配置文件中的字段名无关（配置文件字段名如 curvature、currt_dy 均为历史遗留命名，实际查询变量为速度）。

**表 T1：yawrate_gain_table（速度 → max_theta_deg）**

含义：各速度段允许的最大航向误差等效角（度），用于 Step 6 中 `target_theta` 的幅值上限。
当横向偏差 `lateral_error` 很大时，`atan(dis2lane/prev_dist)` 可能超过此上限，被夹紧。
作用：防止大偏差时产生过激的转向修正。当前全速度段统一为 3.86°。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| max_theta_deg(°) | 3.86 | 3.86 | 3.86 | 3.86 | 3.86 | 3.86 | 3.86 |

**表 T2：theta_yawrate_gain_table（速度 → prev_time_dist）**

含义：预瞄距离时间系数，用于 Step 6 计算 `prev_dist = max(v × prev_time_dist, 5.0m)`。
`prev_dist` 出现在 `atan(dis2lane / prev_dist)` 的分母中——预瞄距离越大，同样横向偏差产生的修正角越小。
作用：决定横向偏差→转角的比例增益（类似 Kp）。当前全速度段 1.5，含义是"前方 1.5 秒处"。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| prev_time_dist | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |

**表 T3：theta_yawrate_gain_table2（速度 → reach_time_theta）**

含义：收敛时间系数，用于 Step 7 的分母 `denom = max(reach_time_theta × v, 3.0m)`。
控制律 `target_κ = -(Δθ + Δθ̇ × T_dt) / denom` 中，`denom` 越大，产生的目标曲率越小。
作用：决定误差→曲率的总增益（类似闭环带宽）。当前全速度段 1.1，含义是"期望在 1.1 秒内收敛"。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| reach_time_theta | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 |

**表 T4：end_pnt_time_table（速度 → T_dt）**

含义：角速度误差项的预测时间（秒），用于 Step 7 控制律 `-(Δθ + Δθ̇ × T_dt) / denom` 中。
`T_dt` 越大，`Δθ̇`（角速度误差）在目标曲率中的权重越高，阻尼效果越强。
作用：提供微分阻尼，抑制航向振荡。低速（0-10 km/h）设为 0 关闭阻尼，≥20 km/h 开启 0.3s。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| T_dt (s) | 0.0 | 0.0 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

**表 T5：dy2heading_time_table（速度 → near_point_time）**

含义：近预瞄点的时间偏移（秒），用于 Step 2 查询 `near = query(t_currt + near_point_time)`。
近预瞄点主要提供 `near.kappa` 写入 `curvature_near`（供外部监控），不直接参与反馈控制律。
当前全速度段 0.1s，即"当前位置往前 0.1 秒"。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| near_point_time (s) | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

**表 T6：dy2_heading_time_coff_table（速度 → far_point_time）**

含义：远预瞄点的时间偏移（秒），用于 Step 2 查询 `far = query(t_currt + far_point_time)`。
远预瞄点的曲率 `far.kappa` 是**前馈路径的核心输入**（Step 9），也参与 `real_dt_theta` 计算（Step 5）。
当前全速度段 1.0s。增大此值可提前预判弯道，但可能在 S 弯中产生提前转向。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| far_point_time (s) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**表 T7：angle_req_max_vlu_table（速度 → max_steer_angle）**

含义：方向盘转角的幅值限制（度），用于 Step 10 合并后的 clamp。
作用：安全保护，防止高速时输出过大转角。低速允许 1100°（几乎不限），≥30 km/h 收紧到 500°。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| max_steer_angle (°) | 1100 | 1100 | 1100 | 500 | 500 | 500 | 500 |

**表 T8：pid_p_param_table（速度 → slip_param）**

含义：侧滑修正增益，用于 Steps 8-9 中 `steer = atan(κ × L) × rad2deg × steer_ratio × slip_param`。
作用：补偿轮胎侧滑角——高速转弯时实际需要的方向盘角度比运动学模型预测值更大。
当前全速度段 1.0（未启用侧滑补偿）。若需补偿高速侧滑，可在高速段设为 >1.0。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| slip_param | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### 2.4 内部状态（帧间持久）

```python
steer_fb_prev    = 0.0  # 上一帧反馈转向角（度），用于反馈通道限制
steer_ff_prev    = 0.0  # 上一帧前馈转向角（度），用于前馈通道限制
steer_total_prev = 0.0  # 上一帧合并转向角（度），用于总速率限制
```

失能时：steer_total_prev 跟踪实际方向盘反馈角。

### 2.5 算法实现

```python
def lat_controller_truck(x, y, yaw_deg, speed_kph, yawrate,
                         steer_feedback, trajectory, ctrl_enable, dt=0.02):
    speed_kph = max(speed_kph, min_speed_prot)
    speed_mps = speed_kph / 3.6
    yaw_rad = yaw_deg * deg2rad

    if not ctrl_enable:
        steer_fb_prev = 0.0
        steer_ff_prev = 0.0
        steer_total_prev = steer_feedback
        return steer_feedback, 0.0, 0.0, 0.0  # 跟随实际方向盘

    # — Step 1：查表查参 ——————————————————————————
    max_theta_deg   = lookup1d(T1, speed_kph)  # 度
    prev_time_dist  = lookup1d(T2, speed_kph)
    reach_time_theta= lookup1d(T3, speed_kph)
    T_dt            = lookup1d(T4, speed_kph)  # 秒
    near_pt_time    = lookup1d(T5, speed_kph)  # 秒
    far_pt_time     = lookup1d(T6, speed_kph)
    max_steer_angle = lookup1d(T7, speed_kph)  # 度
    slip_param      = lookup1d(T8, speed_kph)

    # — Step 2：轨迹查询 ——————————————————————————
    currt = query_nearest_by_position(trajectory, x, y)
    near  = query_nearest_by_relative_time(trajectory,
                currt.relative_time + near_pt_time)
    far   = query_nearest_by_relative_time(trajectory,
                currt.relative_time + far_pt_time)

    # — Step 3：误差计算 ——————————————————————————
    dx = x - currt.x
    dy = y - currt.y
    lateral_error = cos(currt.theta) * dy - sin(currt.theta) * dx  # 米，左正
    heading_error = normalize_angle(yaw_rad - currt.theta)  # rad
    curvature_far = far.kappa

    # — Step 4: calculateRealTheta（kLh=0，退化为 -heading_error）——
    vehicle_speed_clamped = clamp(speed_mps, 1.0, 100.0)
    real_theta = -heading_error - atan(kLh * yawrate / vehicle_speed_clamped)
    # 因此 kLh=0，故 real_theta = -heading_error

    # — Step 5: calculateRealDtTheta ——————————————
    real_dt_theta = -(yawrate - curvature_far * speed_mps)

    # — Step 6: calculateTargetTheta ——————————————
    prev_dist   = max(speed_mps * prev_time_dist, kMin_prev_dist)
    dis2lane    = -lateral_error  # 注意符号：瞬间切入→-lateral_error
    error_angle_raw = atan(dis2lane / prev_dist)
    max_err_angle = min(max_theta_deg * deg2rad, abs(error_angle_raw))
    target_theta = sign(error_angle_raw) * max_err_angle

    target_dt_theta = (sin(real_theta) * speed_mps * prev_dist
                        / (prev_dist**2 + dis2lane**2) * -1.0)

    # — Step 7: calculateTargetCurvature ——————————
    denom = max(reach_time_theta * speed_mps, kMin_reach_dis)
    target_curvature = -((target_theta - real_theta)
                         + (target_dt_theta - real_dt_theta) * T_dt) / denom

    # — Step 8: 反馈转向角 ————————————————————————
    steer_fb_raw = (atan(target_curvature * wheelbase)  # 因已 kLh=0
                     * rad2deg * steer_ratio * slip_param)
    steer_fb = rate_limit(steer_fb_prev, steer_fb_raw, kRate_limit_fb, dt)
    steer_fb_prev = steer_fb

    # — Step 9: 前馈转向角 ————————————————————————
    steer_ff_raw = (atan(curvature_far * wheelbase)
                     * rad2deg * steer_ratio * slip_param)
    steer_ff = rate_limit(steer_ff_prev, steer_ff_raw, kRate_limit_ff, dt)
    steer_ff_prev = steer_ff

    # — Step 10: 合并输出 ————————————————————————
    steer_raw = clamp(steer_fb + steer_ff, -max_steer_angle, max_steer_angle)
    steer_out = rate_limit(steer_total_prev, steer_raw, kRate_limit_total, dt)
    steer_total_prev = steer_out

    return steer_out, currt.kappa, near.kappa, curvature_far
```

## 3. 横向控制器 B：电动牵引车版（LatController）

### 3.1 接口

输入与重卡版相同（见 2.1），输出相同格式。

### 3.2 信号流

```
  ┌──── 查表层 ──────────────────────────────────────────────────────────┐
  │  E1(车速→比例增益Kp)      E2(曲率→预瞄基时间)  E3(车速→预瞄修正)   │
  │  E4(曲率→前馈增益)        E5(车速→前馈增益)                         │
  │  E6(横向偏差→dy转换时间)  E7(横向偏差→dy权重)                       │
  │  E8(横向偏差→航向权重)    E9(曲率→dy转换修正)                       │
  └─────────────────────────────────────────────────────────────────────┘
                              │ (Step 1: 查表 + 预瞄时间计算)
                              ▼
  E2(曲率→基时间) ────> ┌──────────────┐
  E3(车速→修正) ──────> │  预瞄时间计算  │──> end_pnt_time 预瞄时间(IIR滤波)
  上帧预瞄时间 ────────>│  (Step 1)     │    near_time 近预瞄(×0.02)→≈0
                        └──────────────┘    far_time  远预瞄(×0.9)→≈0

  车速 speed_kph ─────> ┌──────────────┐
  速度窗口阈值 ────────>│  速度窗口     │──> spd_window_gain 远预瞄权重(0~1)
  150/200 km/h ────────>│  (Step 2)     │    far_weight = 0.35 × gain
                        └──────────────┘
                           注：重卡/电拖车速远低于 150 km/h
                           → 远预瞄权重恒为 0，仅近预瞄起作用

  车辆位置 x, y ──────> ┌──────────────┐
  规划轨迹 ────────────>│  轨迹查询     │──> currt 当前匹配点
                        │  (Step 3)     │    near  近预瞄点
                        └───┬──────────┘    far   远预瞄点
                            │               ff_pt 前馈点(=当前点)

  车辆航向 yaw_rad ──> ┌───▼──────────┐    E6横向偏差→时间
  轨迹航向 currt.θ ──> │  误差计算      │    E7横向偏差→dy权重
  横向偏差 lat_err ──> │  + 横向偏差    │──> heading_error_near 近预瞄航向误差
                       │    →航向误差   │    heading_error_far  远预瞄航向误差
                       │    转换        │    （含 dy→heading 转换 + IIR 滤波
                       │  (Step 4)      │     + GPS 偏置补偿，当前偏置=0）
                       └───┬──────────┘

  E1 → 比例增益 Kp ──> ┌───▼──────────┐
  积分增益 Ki ────────> │  双预瞄 PID   │──> near_output 近预瞄输出(比例+积分)
  = Kp × 0.125 ──────> │  (Step 5)     │    far_output  远预瞄输出(比例+积分)
                        │  各有独立积分器│    （各自独立限幅）
                        └───┬──────────┘

  近预瞄权重 1.45 ───> ┌───▼──────────┐
  远预瞄权重 ~0 ─────> │  加权合并      │──> angD_DsrdStrAng 反馈期望角
                       │  (Step 6)      │
                       │  + 前馈转角计算 │──> turn_ff_raw 前馈转角
  E4×E5→前馈增益 ────> │  曲率→转角     │    （含速度自适应增益
  特征速度 v_char ───> │  ×速度自适应   │     1+v²/v_char²，v_char=33.3m/s）
                       └───┬──────────┘

                       ┌───▼──────────┐
                       │  转角转换      │──> turn_fb_raw 反馈转角
                       │  (Step 7)      │   = -(期望角 × 转向比 × rad2deg)
                       │  合并: 前馈    │──> steer_raw = 前馈 + 反馈
                       │       +反馈    │
                       └───┬──────────┘

  饱和限幅系数 1.5 ──> ┌───▼──────────┐
  饱和下限 18° ──────> │  饱和限幅      │──> steer_sat 限幅后转角
                       │  (Step 8)      │    上限 = |前馈基准角| × 1.5
                       └───┬──────────┘    下限 = max(该值, 18°)

                       ┌───▼──────────┐
  速率限制 5°/step ──> │  四级输出滤波链│──> steering_target 方向盘转角(°)
                       │  (Step 9)      │
                       │  ① 速率限制    │    防止跳变
                       │  ② 低通1 α=0.01│   平滑高频
                       │  ③ 低通2 α=0.03│   进一步平滑
                       │  ④ 带阻陷波    │    抑制转向柱机械共振
                       └──────────────┘
```

**信号流小结**：
- **预瞄时间自适应**（Step 1）：E2（弯道曲率→基础预瞄时间）× E3（车速→修正系数）→ IIR 平滑 → 近/远预瞄时间。但当前配置下 E2 索引范围（0–60 1/m）与实际道路曲率（≤0.1 1/m）严重不匹配，导致 `end_pnt_time ≈ 0`，预瞄时间退化为极短窗口。这意味着近/远预瞄点几乎重合于当前点。
- **横向偏差→航向误差转换**（Step 4）：将横向位置偏差通过 `atan(横向偏差 / 虚拟距离)` 转换为等效航向误差，叠加到 PID 输入中。作用：让 PID 不仅响应航向偏差，也对横向位置偏差做出修正（类似增加位置反馈通道）。
- **速度窗口**（Step 2）：远预瞄权重根据车速线性插值——150 km/h 以下权重为 0，200 km/h 以上权重为满值。当前重卡/电拖车速远低于 150 km/h，故远预瞄 PID 实际**不参与控制**，只有近预瞄 PID 起作用。
- **有积分器**：近/远预瞄各有独立 PID 积分器（可消除稳态误差），在首次使能和轨迹重规划时清零，防止积分值跨场景携带。
- **四级输出滤波链**（Step 9）：速率限制（防跳变）→ 低通滤波1（平滑）→ 低通滤波2（进一步平滑）→ 带阻陷波（抑制转向柱机械共振频率），逐级处理保证输出平稳。

### 3.3 车辆固定参数（不调整）

| 参数 | 说明 |
|------|------|
| wheelbase | vehicle_param.wheel_base (m) |
| steer_ratio | vehicle_param.steer_ratio |
| max_steer_angle | vehicle_param.max_steer_angle (度) |
| v_char | 120 / 3.6 m/s（速度自适应前馈的特征速度） |

### 3.3 可调控制器参数

**标量参数**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| pid_i_coff_base_p | 0.125 | Ki = Kp × 此值 |
| near_weight_in_angle_req | 1.45 | 近预瞄 PID 输出权重 |
| far_weight_in_angle_req_raw | 0.35 | 远预瞄 PID 输出权重基数 |
| far_coff_in_end_pnt | 0.9 | 远预瞄时间 = end_pnt_time × 此值 |
| near_coff_in_end_pnt | 0.02 | 近预瞄时间 = end_pnt_time × 此值 |
| heading_error_from_dy_gain | 1.0 | 横向偏差→航向误差转换总增益 |
| heading_error_from_dy_filter_coff | 0.15 | dy→heading 转换项的 IIR 系数（直接作为 α） |
| end_pnt_time_filter_coff | 0.1 | 预瞄时间 IIR 滤波系数 |
| lateral_control_use_ff | 1.0 | 前馈总开关（0=纯反馈） |
| angle_req_max_coff_base_ff | 1.5 | 饱和限幅 = ff 基准角 × 此值 |
| angle_req_sat_min_lim | 18.0 度 | 饱和限幅绝对下限 |
| angle_req_rate_lim | 5.0 度/step | 输出速率限制（当前简化为常数） |
| dy2heading_dist_min | 20.0 m | dy→heading 转换的最小虚拟距离 |
| add_heading_err_from_dy_max_lat_err | -1.0 m | 小于此横向误差绝对值时不做 dy→heading（-1 表示始终生效） |
| gps_heading_offset | 0.0 rad | GPS 航向偏置补偿（已设为 0） |
| near_pid_i_max | 3.0 | 近预瞄积分限幅上限（控制律内部） |
| near_pid_i_min | -3.0 | 近预瞄积分限幅下限 |
| near_pid_p_max | 20.0 | 近预瞄比例限幅上限 |
| near_pid_p_min | -20.0 | 近预瞄比例限幅下限 |
| near_pid_i_reset_rate | 0.005 | 近预瞄积分重置速率（× 速度窗口增益） |
| far_pid_i_max | 0.001 | 远预瞄积分限幅上限 |
| far_pid_i_min | -0.001 | 远预瞄积分限幅下限 |
| far_pid_p_max | 0.001 | 远预瞄比例限幅上限 |
| far_pid_p_min | -0.001 | 远预瞄比例限幅下限 |
| far_pid_i_reset_rate | 0.005 | 远预瞄积分重置速率 |
| road_turn_spd_vlu | 150 km/h | 速度窗口低端：低于此远预瞄权重→0 |
| road_normal_lka_spd_vlu | 200 km/h | 速度窗口高端：高于此远预瞄权重→全值 |

**查找表参数**

> LatController 的查找表使用**真实物理量**作为索引（与重卡版不同，字段名即含义）。

**表 E1：pid_p_param_table（speed_kph → Kp 基数）**

作用：Step 5 中近/远预瞄 PID 的比例增益。Ki = Kp × 0.125（pid_i_coff_base_p）。
当前全速度段 Kp=1.0，意味着 PID 的增益完全由 `near_weight`/`far_weight` 和 `steer_ratio` 决定。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| Kp | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**表 E2：end_pnt_time_table（|curvature_near| → 预瞄时间基数）**

作用：Step 1 中预瞄时间的曲率分量。弯道曲率越大，预瞄时间越长以提前响应。
注：当前道路曲率（≤0.1 1/m）远小于表的索引范围（0–60），实际插值始终夹紧在第一点，输出 ≈ 0。
结果：`end_pnt_time ≈ 0`，近/远预瞄退化为 `near_time ≈ 0, far_time ≈ 0`。此表是"占位"状态。

| curvature_near (1/m) | 0.0 | 10.0 | 20.0 | 30.0 | 40.0 | 50.0 | 60.0 |
|---|---|---|---|---|---|---|---|
| base_time (s) | 0.0 | 0.0 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

**表 E3：end_pnt_time_mody_base_spd_table（speed_kph → 预瞄时间速度修正）**

作用：Step 1 中 `end_pnt_time = E2(κ) × E3(v)`，速度越低修正值越小（0.75），减小低速预瞄时间。
但因 E2 输出 ≈ 0，此表实际也未生效。若启用 E2，此表将在低速弯道中缩短预瞄时间。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| modifier | 0.75 | 0.75 | 0.75 | 0.85 | 1.0 | 1.0 | 1.0 |

**表 E4：ang_req_ff_base_curv_table（|κ_current| → 前馈曲率增益分量）**

作用：Step 6 中前馈增益 `ang_req_ff_coff = E4(κ) × E5(v)`。E4 根据当前曲率调整前馈强度。
当前全曲率段 1.0（未区分直道/弯道）。若弯道前馈过强可在大曲率处降低此值。

| curvature (1/m) | 0.001 | 0.002 | 0.004 | 0.01 | 0.02 |
|---|---|---|---|---|---|
| gain | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**表 E5：ang_req_ff_base_spd_table（speed_kph → 前馈速度增益分量）**

作用：Step 6 中前馈增益的速度分量。低速时前馈增益略低（0.88），≥30 km/h 时为 1.0。
补偿低速时运动学模型与实际转向的偏差（低速轮胎侧偏角不同）。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| gain | 0.88 | 0.925 | 0.968 | 1.0 | 1.0 | 1.0 | 1.0 |

**表 E6：dy2heading_time_table（|lateral_error| → dy2heading 时间）**

作用：Step 4 中 `d_min = max(20m, E6(|lat_err|) × E9(κ) × v)` 的时间分量。
`d_min` 是 dy→heading 转换 `atan(lat_err / d_min)` 的分母——时间越大，虚拟距离越大，转换增益越低。
注：横向误差通常 0–2m，索引范围 0–60 导致始终夹紧在第一点，输出 = 0.1。此表实际为常值。

| lateral_error (m) | 0.0 | 10.0 | ... | 60.0 |
|---|---|---|---|---|
| time (s) | 0.1 | 0.1 | 0.1 | 0.1 |

**表 E7：dy_weight_table（|lateral_error| → dy 误差权重）**

作用：Step 4 中 `heading_from_dy_raw = dy_weight × gain × atan(lat_err / d_min)`。
此权重放大横向偏差转换为航向误差的强度。当前全域 2.0，含义是位置偏差的影响力是航向偏差的 2 倍。

| lateral_error (m) | 0.0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| weight | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |

**表 E8：heading_weight_table（|lateral_error| → heading 误差权重）**

作用：Step 4 中 `heading_near_raw = heading_w × normalize_angle(yaw - near.θ)`。
缩放纯航向误差分量。当前全域 1.0（不缩放）。可根据横向偏差大小调整航向与位置的权重分配。

| lateral_error (m) | 0.0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| weight | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**表 E9：dy2_heading_time_coff_table（|curvature_near| → dy2heading 时间修正）**

作用：Step 4 中 `dy2time = E6(|lat_err|) × E9(κ)`，根据曲率修正 dy→heading 的虚拟距离。
注：与 E2 同样的索引范围问题（0–60 vs 实际 ≤0.1），实际始终输出 1.0，此表为占位状态。

| curvature (1/m) | 0.0–60.0 |
|---|---|
| coff | 1.0 |

### 3.4 内部状态（帧间持久）

```python
end_pnt_time_prev      = 1.2    # 预瞄时间（经 IIR 滤波），初始值 1.2s
heading_from_dy_prev   = 0.0    # dy→heading 转换项的 IIR 状态
near_integral          = 0.0    # 近预瞄 PID 积分器
far_integral           = 0.0    # 远预瞄 PID 积分器
steer_prev             = 0.0    # 上一帧输出（用于速率限制）
lpf1_state             = 0.0    # LowPassFilter1 状态（α=0.01）
lpf2_state             = 0.0    # LowPassFilter2 状态（α=0.03）
bandstop_w             = [0.0, 0.0]  # 带阻滤波器状态
curvature_near_prev    = 0.001  # 近预瞄曲率（用于预瞄时间查表）
```

首次使能（ctrl_first_active）时：near_integral = far_integral = 0；
轨迹横向重规划（is_replan）时：同样清零积分器。

### 3.5 算法实现

```python
def lat_controller_electric(x, y, yaw_deg, speed_kph, yawrate,
                            steer_feedback, trajectory,
                            ctrl_enable, ctrl_first_active, is_replan,
                            dt=0.02):
    speed_mps = max(speed_kph, 0.1) / 3.6
    yaw_rad = yaw_deg * deg2rad

    # — Step 1: 查表/速度查数，更新预瞄时间 ——————————
    Kp = lookup1d(E1, speed_kph)
    Ki = Kp * pid_i_coff_base_p  # = Kp × 0.125

    ang_req_ff_curv = lookup1d(E4, abs(current_curvature))  # 用上一帧曲率
    ang_req_ff_spd  = lookup1d(E5, speed_kph)
    ang_req_ff_coff = ang_req_ff_curv * ang_req_ff_spd

    base_time = lookup1d(E2, abs(curvature_near_prev))  # ≈ 0（当前配置）
    spd_modify = lookup1d(E3, speed_kph)
    end_pnt_time_unfilter = base_time * spd_modify
    # IIR 滤波：α = end_pnt_time_filter_coff = 0.1
    end_pnt_time = (end_pnt_time_unfilter * 0.1
                    + end_pnt_time_prev * 0.9)
    end_pnt_time_prev = end_pnt_time

    near_time = end_pnt_time * near_coff_in_end_pnt    # × 0.02
    far_time  = end_pnt_time * far_coff_in_end_pnt     # × 0.9

    # — Step 2: 速度窗口增益（编制远预瞄权重配置）—————
    if speed_kph < road_turn_spd_vlu:    # < 150
        spd_window_gain = 0.0
    elif speed_kph > road_normal_lka_spd_vlu:  # > 200
        spd_window_gain = 1.0
    else:
        spd_window_gain = ((speed_kph - road_turn_spd_vlu)
                           / (road_normal_lka_spd_vlu - road_turn_spd_vlu))
    far_weight = far_weight_in_angle_req_raw * spd_window_gain

    # — Step 3: 查询轨迹点 ————————————————————————
    currt = query_nearest_by_position(trajectory, x, y)
    near  = query_nearest_by_relative_time(trajectory,
                currt.relative_time + near_time)
    far   = query_nearest_by_relative_time(trajectory,
                currt.relative_time + far_time)
    ff_pt = query_nearest_by_relative_time(trajectory,
                currt.relative_time)    # 前馈点 = 当前点

    current_curvature = currt.kappa
    curvature_near    = near.kappa
    curvature_far     = far.kappa
    curvature_near_prev = curvature_near  # 更新内部存储 / 用于预瞄时间查表下帧

    # — Step 4: 误差计算 ——————————————————————————
    dx = x - currt.x
    dy = y - currt.y
    lateral_error = cos(currt.theta) * dy - sin(currt.theta) * dx
    heading_error = normalize_angle(yaw_rad - currt.theta)

    # dy → heading 转换
    dy2time_raw  = lookup1d(E6, abs(lateral_error))    # ≈ 0.1
    dy2time_coff = lookup1d(E9, abs(curvature_near))   # ≈ 1.0
    dy2time      = dy2time_raw * dy2time_coff

    d_min     = max(dy2heading_dist_min,                    # = 20 m
                    dy2time * speed_mps)
    dy_weight = lookup1d(E7, abs(lateral_error))    # ≈ 2.0
    heading_w = lookup1d(E8, abs(lateral_error))    # ≈ 1.0

    heading_from_dy_raw = (dy_weight * heading_error_from_dy_gain
                           * atan(lateral_error / d_min))
    if abs(lateral_error) < add_heading_err_from_dy_max_lat_err:
        heading_from_dy_merge = 0.0
    else:
        heading_from_dy_merge = heading_from_dy_raw

    # IIR 滤波（直接乘系数，非传递函数形式）
    heading_from_dy = (heading_from_dy_merge * heading_error_from_dy_filter_coff
                       + heading_from_dy_prev * (1 - heading_error_from_dy_filter_coff))
    heading_from_dy_prev = heading_from_dy

    # 近/远预瞄航向误差（含 GPS 偏置补偿，当前为 0）
    heading_near_raw = heading_w * normalize_angle(
        yaw_rad + gps_heading_offset - near.theta)
    heading_error_near = normalize_angle(heading_near_raw + heading_from_dy)

    heading_far_raw = heading_w * normalize_angle(
        yaw_rad + gps_heading_offset - far.theta)
    heading_far_from_dy = heading_from_dy * near_coff_in_end_pnt / far_coff_in_end_pnt
    heading_error_far = normalize_angle(heading_far_raw + heading_far_from_dy)

    # — Step 5: Matlab 生成的双预瞄PID ————————————
    # 积分重置
    if ctrl_first_active or is_replan:
        near_integral = far_integral = 0.0

    near_p = clamp(Kp * heading_error_near, near_pid_p_min, near_pid_p_max)
    near_integral = clamp(near_integral + Ki * heading_error_near * dt,
                          near_pid_i_min, near_pid_i_max)
    near_output = near_p + near_integral

    far_p = clamp(Kp * heading_error_far, far_pid_p_min, far_pid_p_max)
    far_integral = clamp(far_integral + Ki * heading_error_far * dt,
                         far_pid_i_min, far_pid_i_max)
    far_output = far_p + far_integral

    # — Step 6: 加权合并 + 前馈转向角 —————————————
    angD_DsrdStrAng = (near_weight_in_angle_req * near_output
                       + far_weight * far_output)

    ff_kappa = curvature_far  # use_far_point_as_ff 已不用
    speed_adapt = 1 + (speed_mps / v_char)**2  # v_char = 120/3.6
    turn_ff_raw = (ang_req_ff_coff
                   * atan(wheelbase * ff_kappa)
                   * speed_adapt
                   * steer_ratio * rad2deg
                   * lateral_control_use_ff)

    # — Step 7: 反馈转向角 ————————————————————————
    turn_fb_raw = -(angD_DsrdStrAng * steer_ratio * rad2deg - 0.0)
    # steer_angle_req_add = 0.0（当前配置）

    steer_raw = turn_ff_raw + turn_fb_raw

    # — Step 8: 饱和限幅 ——————————————————————————
    sat_base = abs(atan(wheelbase * ff_kappa) * steer_ratio * rad2deg)
    sat_max  = max(sat_base * angle_req_max_coff_base_ff, angle_req_sat_min_lim)
    steer_sat = clamp(steer_raw, -sat_max, sat_max)

    # — Step 9: 输出滤波链 ————————————————————————
    # 速率限制（软使能触发，当前 rate_lim 简化为常数 5°/step）
    steer_rl = clamp(steer_sat,
                     steer_prev - angle_req_rate_lim,
                     steer_prev + angle_req_rate_lim)
    # LPF1 (α=0.01)
    steer_lpf1 = steer_rl - 0.01 * lpf1_state
    lpf1_state  = steer_lpf1
    # LPF2 (α=0.03)
    steer_lpf2 = steer_lpf1 - 0.03 * lpf2_state
    lpf2_state  = steer_lpf2
    # 带阻滤波
    steer_out = bandstop_filter.update(steer_lpf2)

    if not ctrl_enable:
        steer_out = steer_feedback    # 跟随实际
    steer_prev = steer_out

    return steer_out, current_curvature, curvature_near, curvature_far
```

## 4. 纵向控制器（LonController）

### 4.1 接口

**输入**

| 变量名 | 来源 | 单位 | 说明 |
|--------|------|------|------|
| x, y | GlobalPose.position_enu | m | 车辆 ENU 坐标 |
| yaw_deg | GlobalPose.euler_angles.z | 度→rad | 航向角 |
| speed_kph | CCANInfoStruct.ccan221_abs_vehspdlgt | km/h | 车速 |
| speed_valid | CCANInfoStruct.ccan221_abs_vehspdlgtstatus | — | 0=有效 |
| accel_mps2 | CCANInfoStruct.ccan242_esp_algt | m/s² | 纵向加速度（加速度计） |
| gear_fb | CCANInfoStruct.ccan123_vcu_displaygear | — | 档位反馈（1-9=D，11=R，其余=N/P） |
| brake_active | CCANInfoStruct.ccan267_ehb_brakeactive | — | 制动踏板状态 |
| curvature_far | cmd.path_curvature_far | 1/m | 横向控制器已写入 |
| trajectory | ADCTrajectory | — | 规划轨迹 |
| ctrl_enable | bool | — | 控制使能 |

**输出**

| 变量名 | 写入字段 | 单位 | 条件 |
|--------|---------|------|------|
| target_torque | cmd.target_torque | Nm | acc_out > 0 且 D 档 |
| brake | cmd.brake | m/s²（负值） | acc_out ≤ 0 |
| gear_location | cmd.gear_location | GEAR_NEUTRAL/DRIVE/PARK | 始终输出 |

### 4.2 信号流

```
  车辆位置/航向/速度 > ┌──────────────┐
  规划轨迹 ──────────> │  Frenet 变换   │──> s_match  沿轨迹弧长
                       │  + 轨迹查询    │    s_dot   沿轨迹速度
                       │  (Step 1)      │    station_error 站位误差(m)
                       └───┬──────────┘        = 参考弧长 - 实际弧长
                           │                speed_error 速度误差(m/s)
                           │                preview_accel_ref 参考加速度
                           │
                       ┌───▼──────────┐
                       │  站位误差保护  │──> station_fnl 保护后站位误差
                       │  (Step 2)     │    低速(≤10km/h)时限制误差
                       └───┬──────────┘    防止静止时误差累积导致起步冲击
                           │
  站位PID ────────────>┌───▼──────────┐
  (Kp=0.25, Ki=0) ───>│  站位 PID     │──> speed_offset 速度补偿(m/s)
  纯比例，无积分 ─────>│  (Step 3)     │    位置偏前→加速，偏后→减速
                       └───┬──────────┘    （外环：位置→速度设定）
                           │
  低速PID(Kp=0.35) ──>┌───▼──────────┐
  高速PID(Kp=0.34) ──>│  速度 PID     │──> acc_closeloop 闭环加速度(m/s²)
  切换阈值3.0 m/s ───>│  (Step 4)     │    速度偏差→加速度指令
  输入限幅 ±5 m/s ───>│  含积分器     │    （内环：速度→加速度）
                       └───┬──────────┘
                           │
  参考加速度(前馈) ──> ┌───▼──────────┐
  前馈权重=1.0 ──────> │  前馈叠加     │──> acc_cmd 总加速度指令
                       │  (Step 5)     │   = 闭环修正 + 轨迹参考加速度
                       └───┬──────────┘    前馈提供基准，PID修正偏差
                           │
  L1(车速→加速度上限)>┌───▼──────────┐
  L2(车速→加速度下限)>│ 最终加速度限制 │──> acc_limited 限幅后加速度
  L3(上帧acc→上升率) >│  (Step 6)     │
  L4(上帧acc→下降率) >│  ①幅值截幅    │    L1/L2 限制绝对范围
  L5(车速→速率增益)  >│  ②速率截幅    │    L3/L4/L5 限制变化速率
  远预瞄曲率 ────────> │  ③急弯收紧    │    弯道(κ<-0.0075)收紧上下限
                       │  ④低速保护    │    极低速额外制动保护
                       └───┬──────────┘
                           │
  IIR 低通 α=0.15 ──> ┌───▼──────────┐
                       │  低通滤波     │──> acc_out 最终加速度(m/s²)
                       └───┬──────────┘    平滑输出，减少执行器振荡
                           │
  档位反馈 gear_fb ──> ┌───▼──────────┐
  车速, 制动状态 ────> │  档位状态机    │──> gear_req 档位请求(N/D/P)
                       │  (Step 7)     │    N→D: 连续50拍(1s)确认
                       └───┬──────────┘    D→N: 连续50拍(1s)确认
                           │               N→P: 连续200拍(4s)确认
                           │
  最终加速度 acc_out ─>┌───▼──────────┐
  实际加速度(传感器) ─>│  扭矩计算     │──> torque_out 驱动扭矩(Nm)
  风阻系数 coef_cd ──> │  (Step 8)     │
  滚阻系数 coef_roll ─>│  仅D档且acc>0 │    动力学模型：
  惯量修正 coef_delta >│              │    风阻+滚阻+坡阻+惯性力
  车辆质量 2440kg ───> │              │    + 加速度跟踪P控制修正
  传动比/效率/轮径 ──> │              │    → 力 × 轮径 / (效率×传动比)
                       └───┬──────────┘
                           │
                       ┌───▼──────────┐
                       │  输出分发     │──> acc>0: 输出(扭矩, 0, 档位)
                       │  (Step 9)     │    acc≤0: 输出(0, 制动, 档位)
                       └──────────────┘
```

**信号流小结**：
- **级联 PID**（Steps 3–4）：外环（站位 PID）将位置偏差转换为速度补偿量，叠加到内环（速度 PID）的设定值上。内环将速度误差转换为加速度指令。这种级联结构将复杂的"位置跟踪"分解为两个简单的单变量控制问题。
- **前馈**（Step 5）：直接使用轨迹规划的参考加速度作为前馈基准——规划说要加速多少就先给多少，PID 只负责修正实际与期望的偏差。前馈使系统响应更快、跟踪更准。
- **双重限幅**（Step 6）：先幅值限制（L1/L2，按车速查表——高速时不允许猛加猛减），再速率限制（L3/L4，按前帧加速度查表——限制 jerk/加速度变化率）。急弯（远预瞄曲率<-0.0075）时额外收紧 25%/40%，保障弯道安全。
- **扭矩模型**（Step 8）：将期望加速度转换为发动机需要输出的扭矩。考虑四种阻力：风阻（与速度²成正比）、滚动阻力（与车重成正比）、坡度阻力（根据历史速度差分估算坡度）、惯性力（加速所需力 = 质量×加速度×惯量修正）。再加上加速度跟踪 P 控制修正（补偿模型误差），最后转换为轮端扭矩。
- **档位状态机**（Step 7）：N（空挡）→ D（前进挡）→ N → P（驻车挡）的状态转换，每次转换需连续确认多拍（1–4秒），防止信号抖动导致误换挡。

### 4.3 车辆固定参数（不调整）

| 参数 | 说明 |
|------|------|
| wheel_rolling_radius | 轮胎半径 (m), vehicle_param |
| transmission_efficiency | 传动效率 η |
| transmission_ratio_D | D 档传动比（约 14.02） |
| transmission_ratio_R | R 档传动比（约 39.8） |
| windward_area | 迎风面积 (m²), vehicle_param |
| veh_mass | 2440 kg（当前硬编码，VCU 重量信号暂未接入） |
| kair_density | 1.2041 kg/m³（空气密度，硬编码） |
| kcoef_gravity | 9.81 m/s²（重力），硬编码 |

### 4.3 可调控制器参数

**PID 增益（最核心）**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| station_pid.kp | 0.25 | 站位误差→速度补偿，比例增益 |
| station_pid.ki | 0.0 | 站位积分（当前关闭） |
| station_pid.integrator_enable | false | — |
| station_pid.sat | 0.3 | 积分限幅 (m/s) |
| low_speed_pid.kp | 0.35 | 低速（< switch_speed）速度 PID Kp |
| low_speed_pid.ki | 0.01 | 低速速度 PID Ki |
| low_speed_pid.sat | 0.3 | 积分限幅 (m/s²) |
| high_speed_pid.kp | 0.34 | 高速速度 PID Kp |
| high_speed_pid.ki | 0.01 | 高速速度 PID Ki |
| high_speed_pid.sat | 0.3 | — |
| switch_speed | 3.0 m/s | 低/高速 PID 切换阈值 |

**前馈与预览参数**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| preview_window | 5.0 拍→0.1 s | 站位/速度预览参考时间 |
| preview_window_for_speed_pid | 50.0 拍→1.0 s | 速度 PID 预览参考时间 |
| acc_cmd_use_preview_point_a | 1.0 | 预览参考加速度前馈权重 |
| station_error_limit | 8.0 m | 站位误差截幅上限 |
| speed_controller_input_limit | 5.0 m/s | 速度 PID 输入截幅 |
| acc_standstill_down_rate | -0.005 m/s²/拍 | 低速（<1.5 m/s）时的加速度下降速率 |

**扭矩模型参数（GFlags，可运行时修改）**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| FLAGS_coef_cd | 0.6 | 风阻系数 |
| FLAGS_coef_rolling | 0.013 | 滚动阻力系数 |
| FLAGS_coef_delta | 1.05 | 转动惯量修正系数（惯性力 = Cδ × m × a） |
| FLAGS_accel_to_torque_kp | 1000 N/(m/s²) | 加速度跟踪 P 控制器增益 |
| FLAGS_torque_combustion_upper_limit | 1200 Nm | 扭矩输出上限 |
| FLAGS_torque_combustion_lower_limit | 0 Nm | 扭矩输出下限 |

**加速度限幅查找表**

**表 L1：acc_up_lim_table（speed_kph → 加速度上限）**

作用：Step 6 中 `acc_clamped = clamp(acc_cmd, L2, L1)` 的上限。
速度越高，允许的最大加速度越小（1.6→1.2 m/s²），保障高速行驶舒适性和安全性。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| acc_up_lim (m/s²) | 1.6 | 1.5 | 1.4 | 1.3 | 1.2 |

**表 L2：acc_low_lim_table（speed_kph → 加速度下限）**

作用：Step 6 中加速度的下限（负值=制动）。低速时制动能力受限（0 km/h 仅 -0.1 m/s²），
防止低速急刹造成的轮胎锁死和乘员不适。高速允许更强制动（-3.5 m/s²）。

| speed (km/h) | 0 | 1 | 2 | 4 | 12 | 25 |
|---|---|---|---|---|---|---|
| acc_low_lim (m/s²) | -0.1 | -0.5 | -1.5 | -2.0 | -3.0 | -3.5 |

**表 L3：acc_up_rate_table（前一帧 acc_out → 加速度上升速率/拍）**

作用：Step 6 中加速度上升速率限制 `acc_up_rate = L3(prev_acc) × L5(spd)`。
限制加速度增大的速度（jerk 限制），防止驱动系统冲击。
注意索引是**前一帧的加速度值**，不是速度——在制动→加速过渡区（-0.1~0.1）速率最低（0.035）。

| prev_acc (m/s²) | -0.5 | -0.25 | -0.1 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|---|---|
| up_rate (m/s²/step) | 0.045 | 0.040 | 0.035 | 0.035 | 0.040 | 0.045 |

**表 L4：acc_down_rate_table（前一帧 acc_out → 加速度下降速率/拍，负值）**

作用：Step 6 中加速度下降速率限制。限制加速度减小（含加速→制动过渡）的速度。
中间区域（-0.1~0.2）速率最低（-0.020），过渡更平缓；两端允许更快变化（-0.030）。

| prev_acc (m/s²) | -1.0 | -0.5 | -0.3 | -0.2 | -0.1 | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| down_rate (m/s²/step) | -0.030 | -0.030 | -0.025 | -0.020 | -0.020 | -0.020 | -0.025 | -0.030 | -0.030 |

**表 L5：acc_rate_gain_table（speed_kph → 速率增益）**

作用：Step 6 中 `acc_up_rate = L3(prev_acc) × L5(spd)`，对上升速率做速度修正。
低速时增益 1.5（允许更快加速变化以提升起步响应），高速时增益 1.0（正常速率）。
注：此增益仅作用于上升速率（L3），不影响下降速率（L4）。

| speed (km/h) | 0 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| gain | 1.5 | 1.5 | 1.35 | 1.2 | 1.0 |

### 4.4 内部状态（帧间持久）

```python
station_pid     = PID(kp=0.25, ki=0.0, ...)      # 站位 PID
speed_pid       = PID(kp=0.35, ki=0.01, ...)      # 速度 PID（参数由高/低速切换）
acc_out_prev    = 0.0   # 上一帧最终输出加速度
iir_acc_state   = 0.0
iir_torque_state = 0.0  # 扭矩低通滤波器状态（当前实现未使用）
speed_history   = [0.0] * 10  # 近10帧速度历史（km/h），用于坡度估计
counter_n2d     = 0     # N→D 计数器（需满50拍）
counter_d2n     = 0
counter_n2p     = 0
torque_integral = 0.0
station_error_fnl_prev = 0.0  # 上一帧最终站位误差
```

首次使能时：重置 station_pid, speed_pid, iir_acc_state,

### 4.5 算法实现

```python
def lon_controller(x, y, yaw_deg, speed_kph, speed_valid, accel_mps2,
                   gear_fb, brake_active, curvature_far,
                   trajectory, ctrl_enable, ctrl_first_active, dt=0.02):
    speed_mps = speed_kph / 3.6
    yaw_rad = yaw_deg * deg2rad

    if ctrl_first_active:
        station_pid.reset()
        speed_pid.reset()
        iir_acc_state = 0.0

    # — Step 1: 计算误差（Frenet 坐标变换）————————————
    matched = query_nearest_by_position(trajectory, x, y)  # PathPoint
    s_match, s_dot, d, d_dot = to_frenet(trajectory, x, y, yaw_rad,
                                          speed_mps, matched)

    t_now   = trajectory.header.measurement_time
    ref_pt  = query_nearest_by_absolute_time(trajectory, t_now)
    prev_pt = query_nearest_by_absolute_time(trajectory,
                t_now + preview_window * dt)                    # +0.1s
    spd_pt  = query_nearest_by_absolute_time(trajectory,
                t_now + preview_window_for_speed_pid * dt)      # +1.0s

    station_error   = ref_pt.s - s_match
    speed_error     = ref_pt.v - s_dot
    preview_speed_error = spd_pt.v - speed_mps
    preview_accel_ref   = prev_pt.a

    # — Step 2: 站位误差保护 ——————————————————————
    station_limited = clamp(station_error, -station_error_limit, station_error_limit)
    # 低速特殊处理（防止静止时因误差过大导致冲击）
    if speed_kph > 10:
        station_fnl = station_limited
    elif station_limited <= 0.25:
        station_fnl = min(0.0, station_limited)
    elif station_limited >= 0.8:
        station_fnl = station_limited
    elif station_error_fnl_prev <= 0.01:
        station_fnl = station_error_fnl_prev
    else:
        station_fnl = station_limited
    station_error_fnl_prev = station_fnl

    # — Step 3: 站位 PID → 速度补偿 ———————————————
    speed_offset = station_pid.control(station_fnl, dt)

    # — Step 4: 速度 PID 切换 + 控制 ——————————————
    if speed_mps <= switch_speed:    # < 3.0 m/s
        speed_pid.set_pid(low_speed_pid.kp, low_speed_pid.ki, 0)
    else:
        speed_pid.set_pid(high_speed_pid.kp, high_speed_pid.ki, 0)

    speed_input = clamp(speed_offset + preview_speed_error,
                        -speed_controller_input_limit,
                        speed_controller_input_limit)
    acc_closeloop = speed_pid.control(speed_input, dt)

    # — Step 5: 前馈叠加（坡度补偿当前实际未生效）————————
    acc_cmd = acc_closeloop + acc_cmd_use_preview_point_a * preview_accel_ref
    # slope_compensation ≈ 0（grade_percent 硬编码为 0）

    # — Step 6: CalFinalAccCmd — 幅值 + 速率双重限制 ———
    if ctrl_enable:
        acc_up_lim  = lookup1d(L1, abs(speed_kph))
        acc_low_lim = lookup1d(L2, abs(speed_kph))
        acc_up_rate_raw = lookup1d(L3, acc_out_prev)
        acc_dn_rate_raw = lookup1d(L4, acc_out_prev)
        rate_gain   = lookup1d(L5, abs(speed_kph))
        acc_up_rate = acc_up_rate_raw * rate_gain

        # 急弯收紧（curvature_far < -0.0075 时）
        if curvature_far < -0.0075:
            acc_up_lim  *= 0.75
            acc_low_lim *= 0.60

        # 低速保护：下降速率使用静止值
        if abs(speed_mps) < 1.5:
            acc_dn_rate = acc_standstill_down_rate    # = -0.005
        else:
            acc_dn_rate = acc_dn_rate_raw

        # 幅值截幅
        acc_clamped = clamp(acc_cmd, acc_low_lim, acc_up_lim)

        # 低速额外下限保护
        if abs(speed_mps) >= 0.2 or acc_clamped >= 0.25:
            acc_lowspd = acc_clamped
        else:
            acc_lowspd = min(-0.05, acc_clamped)

        # 速率截幅
        acc_limited = clamp(acc_lowspd,
                            acc_out_prev + acc_dn_rate,
                            acc_out_prev + acc_up_rate)
        acc_out_prev = acc_limited
    else:
        acc_limited = 0.0
        acc_out_prev = 0.0

    # IIR 低通（α=0.15）：y = x - 0.15 × y_prev
    acc_out = acc_limited - 0.15 * iir_acc_state
    iir_acc_state = acc_out

    # — Step 7: 档位状态机（GearControl）——————————
    # D 档判断：gear_fb 在 1-9 范围内
    is_D = (1 <= gear_fb <= 9)
    is_R = (gear_fb == 11)
    current_gear_enum = 'D' if is_D else ('R' if is_R else 'N')

    # N→D
    if (current_gear_enum == 'N' and acc_cmd > 0.01
            and ctrl_enable and speed_mps <= 0.5 and speed_valid == 0):
        counter_n2d += 1
    else:
        counter_n2d = 0
    if counter_n2d >= 50 and speed_mps <= 0.25:
        gear_req = 'D'
        counter_n2d = 0

    # D→N
    elif (current_gear_enum == 'D' and acc_cmd < 0.001
          and speed_mps <= 0.01 and ctrl_enable):
        counter_d2n += 1
        if counter_d2n >= 50:
            gear_req = 'N'
            counter_d2n = 0

    # N→P
    elif (current_gear_enum == 'N' and ctrl_enable
          and acc_cmd < 0.001 and speed_mps <= 0.01 and speed_valid == 0):
        if not brake_active:
            counter_n2p += 1
        else:
            counter_n2p = 0
        if counter_n2p >= 200:
            gear_req = 'P'
            counter_n2p = 0

    else:
        gear_req = current_gear_enum

    # — Step 8: CalFinalTorque — 扭矩模型（仅 D 档且 acc_out > 0）——
    torque_out = 0.0
    if is_D and acc_out > 0:
        # 坡度估算（10 帧速度差分）
        speed_history.pop()
        speed_history.insert(0, speed_mps)
        if len(history_valid_frames) >= 10:
            dvdt = sum(speed_history[i] - speed_history[i+5]
                       for i in range(5)) / (25 * dt)    # 5 组差分均值
            road_slope = (accel_mps2 - dvdt / kcoef_gravity) / kcoef_gravity
        else:
            road_slope = 0.0

        F_air     = 0.5 * FLAGS_coef_cd * kair_density * windward_area * speed_mps**2
        F_rolling = FLAGS_coef_rolling * veh_mass * kcoef_gravity * cos(road_slope)
        F_slope   = veh_mass * kcoef_gravity * sin(road_slope)
        F_inertia = FLAGS_coef_delta * veh_mass * acc_out
        F_resist  = F_air + F_rolling + F_slope + F_inertia

        # 加速度跟踪 P 控制（注：积分项被累计但当前实现中未加入输出）
        error_ax = acc_out - accel_mps2
        F_P = FLAGS_accel_to_torque_kp * error_ax

        trans_ratio = transmission_ratio_D
        T_raw = (F_resist + F_P) * wheel_rolling_radius / (transmission_efficiency * trans_ratio)
        torque_out = clamp(T_raw,
                           FLAGS_torque_combustion_lower_limit,
                           FLAGS_torque_combustion_upper_limit)

        if not ctrl_enable:
            torque_out = 0.0

    # — Step 9: 输出分发 ——————————————————————————
    if acc_out > 0:
        return torque_out, 0.0, gear_req    # torque, brake, gear
    else:
        return 0.0, acc_out, gear_req        # brake = acc_out（负值）
```

## 5. 参数快速索引

### 5.1 固定车辆参数（两个控制器共用，不参与调参）

| 参数名 | 来源 | 含义 |
|--------|------|------|
| wheelbase | vehicle_param.wheel_base | 轴距 (m) |
| steer_ratio | vehicle_param.steer_ratio | 转向比 |
| wheel_rolling_radius | vehicle_param.wheel_rolling_radius | 轮胎半径 (m) |
| transmission_efficiency | vehicle_param.transmission_efficiency | 传动效率 |
| transmission_ratio_D/R | vehicle_param | 档位传动比 |
| windward_area | vehicle_param.windward_area | 迎风面积 (m²) |

### 5.2 可微调控制器参数汇总

**横向（重卡）**：表 T1–T8 的各节点 value（共 8 张表，全部以 speed_kph 为索引）

| 表 | 信号流角色 | 当前状态 |
|----|-----------|---------|
| T1 | 横向偏差→转角的幅值上限 | 全速 3.86°（统一） |
| T2 | 预瞄距离→比例增益（类 Kp） | 全速 1.5s（统一） |
| T3 | 收敛时间→闭环带宽 | 全速 1.1s（统一） |
| T4 | 角速度阻尼权重 | 0-10 km/h 关闭，≥20 开启 0.3s |
| T5 | 近预瞄点时间偏移（不直接参与反馈控制律） | 全速 0.1s |
| T6 | 远预瞄点时间偏移（**前馈核心输入**） | 全速 1.0s |
| T7 | 输出转角安全限幅 | 低速 1100°，≥30 km/h 500° |
| T8 | 侧滑修正增益 | 全速 1.0（未启用） |

**横向（电动牵引车）**：

- 标量：Kp（通过 E1），Ki = Kp×0.125，near_weight，far_weight_raw，near/far_coff，heading_from_dy_gain，angle_req_max_coff_base_ff

| 表 | 信号流角色 | 当前状态 |
|----|-----------|---------|
| E1 | PID 比例增益基数 | 全速 1.0（统一） |
| E2 | 预瞄时间的曲率分量 | **占位**（索引范围不匹配，输出≈0） |
| E3 | 预瞄时间的速度修正 | 低速 0.75→高速 1.0（但因 E2≈0 未生效） |
| E4 | 前馈增益的曲率分量 | 全曲率 1.0（统一） |
| E5 | 前馈增益的速度分量 | 低速 0.88→≥30 km/h 1.0 |
| E6 | dy→heading 转换的虚拟距离时间 | 全域 0.1（常值） |
| E7 | 横向偏差→航向误差的权重（≈位置增益） | 全域 2.0 |
| E8 | 纯航向误差的权重 | 全域 1.0 |
| E9 | dy→heading 的曲率修正 | **占位**（索引范围不匹配，输出≡1.0） |

**纵向**：

- 标量：station_kp，low_kp/ki，high_kp/ki，switch_speed，acc_cmd_use_preview_point_a，FLAGS_accel_to_torque_kp，FLAGS_coef_cd/rolling/delta

| 表 | 信号流角色 | 当前状态 |
|----|-----------|---------|
| L1 | 加速度上限（速度相关） | 1.6→1.2 m/s²（速度越高越低） |
| L2 | 加速度下限/最大制动（速度相关） | -0.1→-3.5 m/s²（低速限制制动） |
| L3 | 加速度上升速率（jerk上限） | 0.035~0.045 m/s²/step |
| L4 | 加速度下降速率（jerk下限） | -0.020~-0.030 m/s²/step |
| L5 | 上升速率的速度增益 | 低速 1.5→高速 1.0 |

### 5.3 不参与调参的量

| 类别 | 例子 | 原因 |
|------|------|------|
| 物理/传感信号 | 车速、位置、加速计、航向角 | 输入，非设计量 |
| 车辆固定常数 | 轴距、传动比、轮胎半径 | 由硬件决定 |
| 底盘标定表 | calibration_table(speed, acc → torque_cmd) | 描述底盘非线性，由实车标定 |
| 硬编码安全约束 | kRate_limit_fb/ff/total、带阻滤波器系数 | 稳定性/安全保障，不参与稳态优化 |
| 档位状态机阈值 | N→D 计数器阈值 50 拍 | 安全逻辑 |
