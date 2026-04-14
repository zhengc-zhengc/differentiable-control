# 控制器算法实现规格书（v2）

v2 文档改了什么

文件：src/mlp/docs/controller_spec_v2.md（原文件不动）

横向控制器（重卡版） - 2 处算法修正

+---+--------------+----------------+----------------------------------+
| # | 位置         | v1 文档（错误）| v2 文档（正确）                  |
+---+--------------+----------------+----------------------------------+
| 1 | Step 6       | = sign(error) *| = -sign(error) *                 |
|   | target_theta |   max_err      |   max_err（代码 line 399 乘了    |
|   |              |                |   -1.0）                         |
+---+--------------+----------------+----------------------------------+
| 2 | 失能时输出   | return         | return 0.0（代码输出             |
|   |              | steer_feedback | 0，last1step_angle_              |
|   |              |                | 跟踪实际用于平滑过渡）           |
+---+--------------+----------------+----------------------------------+

纵向控制器 - 9 处修改

+---+--------------+----------------+----------------------------------+
| # | 类别         | v1 文档        | v2 文档                          |
+---+--------------+----------------+----------------------------------+
| 3 | 站位误差     | speed_kph > 10 | speed_kph > 1.0                  |
|   | 保护阈值     |                |                                  |
+---+--------------+----------------+----------------------------------+
| 4 | 低速起步     | acc >= 0.25    | acc >=                           |
|   | acc 阈值     |                | 0.06（kMinAccForStarting）       |
+---+--------------+----------------+----------------------------------+
| 5 | 扭矩/制动    | acc_out > 0    | acc_out >                        |
|   | 死区         |                | -0.05（kAccelerationDeadZone）   |
+---+--------------+----------------+----------------------------------+
| 6 | 低通滤波     | 单个           | 两个**独立** IIR(α=0.15)：扭矩通道一 |
|   | 器结构       | IIR(α=0.15)    | 个、制动通道一个，互斥 Reset     |
|   |              | 在分支前       |                                  |
+---+--------------+----------------+----------------------------------+
| 7 | 预览速度     | 直接用 spd_pt.v| 低速截断 + IIR α=0.1 滤波        |
|   | 参考         |                |                                  |
+---+--------------+----------------+----------------------------------+
| 8 | 预览加速     | 直接用         | IIR 滤波，系数                   |
|   | 度参考       | prev_pt.a      | a_preview_point_filt_coff=0.05   |
+---+--------------+----------------+----------------------------------+
| 9 | Step 5       | 硬编码 0       | 补录 VCU 坡角信号接入逻辑（FLAGS |
|   | 坡度补偿     |                | _enable_slope_offset，当前=false |
|   |              |                | 不生效）                         |
+---+--------------+----------------+----------------------------------+
| 10| 车辆质量     | 2440 kg 硬编码 | 默认 9300 kg，VCU 动态读取（1-55 |
|   |              |                | 吨范围）                         |
+---+--------------+----------------+----------------------------------+
| 11| 档位映射     | N/P=其他，D=1-9| N=0，R=1，D=2-5；N→D 阈值        |
|   | + 状态机     | ，R=11；N→D    | 0.06，D→N 速度 0.1 m/s；新增 P→N |
|   |              | 阈值 0.01，D→N | 转换                             |
|   |              | 速度 0.01 m/s  |                                  |
+---+--------------+----------------+----------------------------------+

**版本说明**：v1 基于非脱敏的视频回放；v2 于 2026-04-13 结合 `lat_controller_truck.cc` 与 `lon_controller.cc` 当前代码及 `control_conf_truck.pb.txt`/`control.conf` 全面校对，修正了若干达场景错误识别并补录子项设计细节。

目的：根据此文档可让 Python 工程师直接用完整规格填入控制器，整编出基于 C++ 原始代码的物理信号的数据及等等的。

执行频率：50 Hz，步长 dt = 0.02 s

车型选择：use_truck_lat_control = true -> 重卡横向控制器；false = 电动牵引车横向控制器。纵向控制器两者共用。

## 0. 符号约定

| 符号 | 含义 |
|------|------|
| clamp(x, lo, hi) | max(lo, min(hi, x)) |
| normalize_angle(a) | 折叠回归一化到 [-π, π] |
| sign(x) | +1/0/-1，\|x\| < 1e-3 时取 0 |
| deg2rad | π / 180 |
| rad2deg | 180 / π |
| [prev] | 上一控制周期的值（内部状态） |

## 0.1 系统信号流总览

```
+------------------------------------------------------------------+
|           ControlComponent（定时组件，50Hz 主循环）              |
|                                                                  |
|  GlobalPose   ---->   横向控制器    ---->   方向盘转角指令        |
|  （位置/航向）        （重卡版 or 电拖版）  steering_target (°)   |
|                                                                  |
|  CCANInfoStruct -->                  ---->  远预瞄曲率            |
|  （车速/横摆率等）                          curvature_far         |
|                                                                  |
|  ADCTrajectory  -->                                              |
|  （规划轨迹）    |--> 纵向控制器     <----                        |
|                      LonController                               |
|                      读取 curvature_far（横↔纵唯一耦合信号）      |
|                                                                  |
|                                    ---> 驱动扭矩 target_torque (Nm，加速时) |
|                                    ---> 制动减速度 brake (m/s²，减速时)    |
|                                    ---> 档位请求 gear_location (N/D/P)     |
+------------------------------------------------------------------+
```

**执行顺序**：横向控制器先执行，将远预瞄曲率 `path_curvature_far` 写入 ControlCommand；
纵向控制器后执行，读取该曲率用于急弯减速判断。信息流**单向**（横→纵向），
两个控制器之间无反向耦合。

## 1. 公共基础组件

### 1.1 Lookup1D 分段线性插值

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

代码中使用 `{num=[0,0,1], den=[0,α,1]}` 的传递函数，等价于：

```python
class IIR:
    def __init__(self, alpha: float):
        self.alpha = alpha  # 时间常数 τ（秒），实际增益 = dt/(τ+dt)
        self.y_prev = 0.0
    def update(self, x: float) -> float:
        # 等效：y = x - alpha * y_prev（den 格式的递推式）
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
    # 离散迭代（标准 Direct Form II Transposed）：
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

| 字段          | 单位 | 含义                  |
|---------------|------|-----------------------|
| x, y          | m    | ENU 坐标              |
| theta         | rad  | 切线方向（航向角）    |
| kappa         | 1/m  | 路径曲率（左转为正）  |
| v             | m/s  | 参考速度              |
| a             | m/s² | 参考加速度            |
| s             | m    | 累积弧长              |
| relative_time | s    | 相对轨迹起点的时间    |
| absolute_time | s    | 绝对时间戳            |

```python
def query_nearest_by_position(traj, x, y):
    """返回距离 (x,y) 欧氏距离最近的轨迹点。"""

def query_nearest_by_relative_time(traj, t_rel):
    """按 relative_time 线性插值，超界夹紧到端点。"""

def query_nearest_by_absolute_time(traj, t_abs):
    """按 absolute_time 线性插值,超前则当前采到起点。"""

def to_frenet(traj, x, y, theta_rad, v_mps, matched_point):
    """Frenet 坐标变换,返回 (s_matched, s_dot, d, d_dot)。"""
    heading_err      = normalize_angle(theta_rad - matched_point.theta)
    s_dot            = v * cos(heading_err)
    d                = cos(matched_point.theta)*(y - matched_point.y)
                     - sin(matched_point.theta)*(x - matched_point.x)
    d_dot            = v * sin(heading_err)
    s_matched        = matched_point.s + projection_along_path
```

## 2. 横向控制器 A:重卡版(LatControllerTruck)

### 2.1 接口

**输入**

| 变量名        | 来源                                         | 单位    | 说明                                   |
|---------------|----------------------------------------------|---------|----------------------------------------|
| x, y          | GlobalPose.position_enu                      | m       | 车辆 ENU 坐标                          |
| yaw_deg       | GlobalPose.euler_angles.z                    | 度→rad  | 车辆航向角,需转弧度                    |
| speed_kph     | CCANInfoStruct.ccan221_abs_vehspdlgt         | km/h    | 车速                                   |
| yawrate       | CCANInfoStruct.ccan242_esp_yawrate           | rad/s   | 横摆角速度                             |
| steer_feedback| CCANInfoStruct.ccan175_eps1_steeringwheelangle | 度    | 方向盘实际角度(仅用于失能时回脱)      |
| trajectory    | ADCTrajectory                                |         | 规划输出点列表                         |
| ctrl_enable   | bool                                         |         | 控制使能                               |

**输出**

| 变量名            | 写入字段                   | 单位                   |
|-------------------|----------------------------|------------------------|
| steering_target   | cmd.steering_target        | 度                     |
| curvature_current | cmd.path_curvature_current | 1/m                    |
| curvature_near    | cmd.path_curvature_near    | 1/m                    |
| curvature_far     | cmd.path_curvature_far     | 1/m(纵向控制器读取)   |

### 2.2 信号流

```
查表层 (全以 speed_kph 车速为索引)
+-----------------------------------------------------------------+
| T1→最大误差角    T2→预瞄距离时间    T3→收敛时间    T4→阻尼时间   |
| T5→近预瞄时间    T6→远预瞄时间     T7→最大转角   T8→侧滑修正    |
+-----------------------------------------------------------------+
                                 |
                              (Step 1: 查表)
                                 v
车辆位置 x,y   ----------->   +-------------+    curr 当前匹配点(位置最近)
规划轨迹 trajectory  ------>  | 轨迹查询    | -> near 近预瞄点(+0.1s)
                              |  (Step 2)   |    far 远预瞄点(+1.0s)
                              +-------------+

车辆航向 yaw_rad  ----------->  +-------------+    lateral_error  横向偏差(m,左正)
轨迹航向 curr.θ   ----------->  | 误差计算    | ->  heading_error  航向误差(rad)
                                |  (Step 3)   |    curvature_far  远点曲率(1/m)
                                +-------------+

横摆角速度 yawrate  -------->   +-------------+
车速 speed_mps  ----------->    | 实际航向角  | ->  real_theta  实际航向误差
                                |  (Step 4)   |    = -heading_error
                                +-------------+    (kLh=0 铰接修正退化为恒等)

横摆角速度 yawrate  -------->
远点曲率 K_far  ----------->    +-------------+
车速 speed_mps  ----------->    | 航向角变化率| ->  real_dt_theta  航向误差变化率
                                |  (Step 5)   |    = -(ω - K_far × v)
                                +-------------+

T1 最大误差角  ----------->
T2 预瞄距离   ----------->     +-------------+
横向偏差 lat_err  ---------->   | 目标航向角  | ->  target_theta     期望航向(限幅后)
                                |  (Step 6)   |    target_dt_theta   期望变化率
                                +-------------+

T3 收敛时间   ----------->
T4 阻尼时间   ----------->     +-------------+
                               | 目标曲率    | ->  target_curvature  反馈曲率
                               |  (Step 7)   |    = -(Δθ + Δθ×T_dt) / 收敛距离
                               +-------------+

T8 侧滑修正  ----------->     +-------------+  +-------------+
轴距 wheelbase  ----------->   | 反馈转角    |  | 前馈转角    | <--  远点曲率 K_far
转向比 steer_ratio  ------->   |  (Step 8)   |  |  (Step 9)   |   (道路曲率直接补偿)
                               +-------------+  +-------------+
                               速率限制 120°/s  速率限制 165°/s

T7 最大转角  ----------->     +-------------+
                               | 合并 + 限幅 | ->  steering_target  方向盘转角(°)
                               |  (Step 10)  |    总速率限制 300°/s
                               +-------------+
```

**信号流小结**:
- **反馈路径**(Steps 3-8):车辆偏离轨迹的位置/航向误差 → 目标曲率 → 反馈转角。T2(预瞄距离)控制比例增益,T1(最大误差角)限幅输出。
- **前馈路径**(Step 9):远预瞄点的道路曲率"K_far"直接转换为前馈转角,用于补偿弯道——即使车辆没有偏差,弯道中也需要转向。
- **合并**(Step 10):反馈 + 前馈 → 幅值限幅(T7)→ 总速率限制(防止方向盘跳变)→ 输出。
- **无积分器**:全部纯比例/纯积分,不累积历史误差。优点是无超调风险,缺点是响应有稳态误差。

### 2.3 车辆固定参数(不调整)

| 参数          | 车型值                       | 说明                          |
|---------------|------------------------------|-------------------------------|
| wheelbase     | vehicle_param.wheel_base (m) | 轴距                          |
| steer_ratio   | vehicle_param.steer_ratio    | 转向传动比(方向盘角/车轮转角) |
| min_speed_mps | 0.1 km/h                     | 速度保护下限                  |

### 2.4 **硬编码常量(代码级,不在配置文件)**

| 常量             | 值       | 含义                             |
|------------------|----------|----------------------------------|
| kLh              | 0.0 m    | 铰接轴距(保留接口,当前为 0)      |
| kRate_limit_fb   | 120 °/s  | 反馈通道速率限制                 |
| kRate_limit_ff   | 165 °/s  | 前馈通道速率限制                 |
| kRate_limit_total| 300 °/s  | 合并后总速率限制                 |
| kMin_prev_dist   | 5.0 m    | 预瞄距离最小值                   |
| kMin_reach_dis   | 3.0 m    | 收敛距离最小值                   |

### 2.5 可调控制器参数(查找表)

重要说明:以下所有查找表在代码中均以**当前车速(km/h)**作为查询索引,与配置文件中的字段名无关。配置文件字段名如 curvature、curr_dy 均为历史遗留命名,实际查询变量都是车速。

**表 T1: yawrate_gain_table(速度 → max_theta_deg)**

含义:全速度段允许的最大航向误差等效角(度),用于 Step 6 中 `target_theta` 的幅值上限。
当横向偏差 `lateral_error` 很大时,`atan(dis2lane/prev_dist)` 可能超出此上限,被夹紧。
作用:防止大偏差时产生过激的转向修正。当前全速度段统一为 3.86°。

| speed (km/h)    | 0    | 10   | 20   | 30   | 40   | 50   | 60   |
|-----------------|------|------|------|------|------|------|------|
| max_theta_deg   | 3.86 | 3.86 | 3.86 | 3.86 | 3.86 | 3.86 | 3.86 |

**表 T2: theta_yawrate_gain_table(速度 → prev_time_dist)**

含义:预瞄距离时间系数,用于 Step 6 计算 `prev_dist = max(v × prev_time_dist, 5.0m)`。
`prev_dist` 出现在 `atan(dis2lane / prev_dist)` 的分母中——预瞄距离越大,同样横向偏差产生的修正角越小。
作用:决定横向偏差→转角的比例增益(类似 Kp)。当前全速度段 1.5,含义是"前方 1.5 秒处"。

| speed (km/h)    | 0   | 10  | 20  | 30  | 40  | 50  | 60  |
|-----------------|-----|-----|-----|-----|-----|-----|-----|
| prev_time_dist  | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |

**表 T3: theta_yawrate_gain_table2(速度 → reach_time_theta)**

含义:收敛时间系数,用于 Step 7 的分母 `denom = max(reach_time_theta × v, 3.0m)`。
控制律 `target_k = -(Δθ + Δθ × T_dt) / denom` 中 `denom` 越大,产生的目标曲率越小。
作用:决定误差→曲率的总增益(类似闭环补带宽)。当前全速度段 1.1,含义是"期望在 1.1 秒内收敛"。

| speed (km/h)     | 0   | 10  | 20  | 30  | 40  | 50  | 60  |
|------------------|-----|-----|-----|-----|-----|-----|-----|
| reach_time_theta | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 |

**表 T4: end_pnt_time_table(速度 → T_dt)**

含义:角速度误差加预瞄时间(秒),用于 Step 7 控制律 `-(Δθ + Δθ × T_dt) / denom` 中。
`T_dt` 越大,`Δθ`(角速度误差)在目标曲率中的权重越高,阻尼效果越强。
作用:提供微分阻尼,当前设为 0 关闭阻尼,≥20 km/h 开启 0.3s。

| speed (km/h)  | 0   | 10  | 20  | 30  | 40  | 50  | 60  |
|---------------|-----|-----|-----|-----|-----|-----|-----|
| T_dt (s)      | 0   | 0   | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

**表 T5: dy2heading_time_table(速度 → near_point_time)**

含义:近预瞄点的时间偏移(秒),用于 Step 2 查询 `near = query(t_curr + near_point_time)`。
近预瞄点主要提供 `near.kappa` 写入 `curvature_near`(供外部监控),不直接参与反馈控制律。
当前全速度段 0.1s,即"当前位置往前 0.1 秒"。

| speed (km/h)       | 0   | 10  | 20  | 30  | 40  | 50  | 60  |
|--------------------|-----|-----|-----|-----|-----|-----|-----|
| near_point_time (s)| 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

**表 T6: dy2_heading_coff_table(速度 → far_point_time)**

含义:远预瞄点的时间偏移(秒),用于 Step 2 查询 `far = query(t_curr + far_point_time)`。
远预瞄点的曲率 `far.kappa` 是**前馈路径的核心输入**(Step 9),也参与 `real_dt_theta` 计算(Step 5)。
当前全速度段 1.0s,增大此值可提前预瞄弯道,用于 S 弯中产生提前转向。

| speed (km/h)       | 0   | 10  | 20  | 30  | 40  | 50  | 60  |
|--------------------|-----|-----|-----|-----|-----|-----|-----|
| far_point_time (s) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**表 T7: angle_req_max_vlu_table(速度 → max_steer_angle)**

含义:方向盘转角的幅值限制(度),用于 Step 10 合并后的 clamp。
作用:安全保护,防止高速时输出过大转角。低速允许 1100°(几乎不限),≥30 km/h 收紧到 500°。

| speed (km/h)      | 0    | 10   | 20   | 30   | 40   | 50   | 60   |
|-------------------|------|------|------|------|------|------|------|
| max_steer_angle(°)| 1100 | 1100 | 1100 | 500  | 500  | 500  | 500  |

**表 T8: pid_p_param_table(速度 → slip_param)**

含义:侧滑修正增益,用于 Steps 8-9 中 `steer = atan(κ × L) × rad2deg × steer_ratio × slip_param`。
作用:补偿轮胎侧偏——高速转弯时实际车辆角比运动学模型预测值更大。
当前全速度段 1.0(未启用侧滑补偿),若需补偿高速侧滑,只在高速段设为 >1.0。

| speed (km/h) | 0   | 10  | 20  | 30  | 40  | 50  | 60  | 70  | 120 |
|--------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| slip_param   | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### 2.4 内部状态(帧间持久)

```python
steer_fb_prev    = 0.0   # 上一帧反馈转向角(度),用于反馈通道限速
steer_ff_prev    = 0.0   # 上一帧前馈转向角(度),用于前馈通道限速
steer_total_prev = 0.0   # 上一帧合并转向角(度),用于总速率限制
```

失能时:steer_total_prev 跟踪实际方向盘反馈角。

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
        steer_total_prev = steer_feedback  # 跟踪实际方向盘（用于重新使能时的平滑过渡）
        return 0.0, 0.0, 0.0, 0.0  # 失能时输出 0，由底层硬件接管方向盘

    # - Step 1：查表查参
    max_theta_deg    = lookup1d(T1, speed_kph)  # 度
    prev_time_dist   = lookup1d(T2, speed_kph)
    reach_time_theta = lookup1d(T3, speed_kph)
    T_dt             = lookup1d(T4, speed_kph)  # 秒
    near_pt_time     = lookup1d(T5, speed_kph)  # 秒
    far_pt_time      = lookup1d(T6, speed_kph)
    max_steer_angle  = lookup1d(T7, speed_kph)  # 度
    slip_param       = lookup1d(T8, speed_kph)

    # - Step 2：轨迹查询
    curr = query_nearest_by_position(trajectory, x, y)
    near = query_nearest_by_relative_time(trajectory,
                curr.relative_time + near_pt_time)
    far  = query_nearest_by_relative_time(trajectory,
                curr.relative_time + far_pt_time)

    # - Step 3：误差计算
    dx = x - curr.x
    dy = y - curr.y
    lateral_error = cos(curr.theta) * dy - sin(curr.theta) * dx  # 米，左正
    heading_error = normalize_angle(yaw_rad - curr.theta)  # rad
    curvature_far = far.kappa

    # - Step 4：calculateRealTheta（kLh=0，退化为 -heading_error）---
    vehicle_speed_clamped = clamp(speed_mps, 1.0, 100.0)
    real_theta = -(heading_error + atan(kLh * yawrate / vehicle_speed_clamped))
    # 因此 kLh=0，故 real_theta = -heading_error

    # - Step 5：calculateRealDtTheta ---
    real_dt_theta = -(yawrate - curvature_far * speed_mps)

    # - Step 6：calculateTargetTheta ---
    prev_dist = max(speed_mps * prev_time_dist, kMin_prev_dist)
    dis2lane = -lateral_error  # 注意符号：瞬间切入+-lateral_error

    error_angle_raw = atan(dis2lane / prev_dist)
    max_err_angle = min(max_theta_deg * deg2rad, abs(error_angle_raw))
    # 注意：target_theta 取 **负** sign（v1 文档此处符号写反）
    # 代码 calculateTargetTheta 内部即以 -1.0，保证车辆偏左时产生负曲率（右转修正）
    target_theta = -sign(error_angle_raw) * max_err_angle

    target_dt_theta = (sin(real_theta) * speed_mps * prev_dist
                       / (prev_dist**2 + dis2lane**2) * 1.0)

    # - Step 7: calculateTargetCurvature
    denom = max(reach_time_theta * speed_mps, kMin_reach_dis)
    target_curvature = -((target_theta - real_theta)
                         + (target_dt_theta - real_dt_theta) * T_dt) / denom

    # - Step 8: 反馈转向角
    steer_fb_raw = (atan(target_curvature * wheelbase)  # 因 kLh=0
                    * rad2deg * steer_ratio * slip_param)
    steer_fb = rate_limit(steer_fb_prev, steer_fb_raw, kRate_limit_fb, dt)
    steer_fb_prev = steer_fb

    # - Step 9: 前馈转向角
    steer_ff_raw = (atan(curvature_far * wheelbase)
                    * rad2deg * steer_ratio * slip_param)
    steer_ff = rate_limit(steer_ff_prev, steer_ff_raw, kRate_limit_ff, dt)
    steer_ff_prev = steer_ff

    # - Step 10: 合并输出
    steer_raw = clamp(steer_fb + steer_ff, -max_steer_angle, max_steer_angle)
    steer_out = rate_limit(steer_total_prev, steer_raw, kRate_limit_total, dt)
    steer_total_prev = steer_out

    return steer_out, curr.kappa, near.kappa, curvature_far
```

## 3. 横向控制器 B：电动牵引车版（LatController）

### 3.1 接口

输入与重卡版相同（见 2.1），输出相同格式。

### 3.2 信号流

> **说明**：Section 3.2 及以下（原文档 L519-894，约 376 行）**有意省略**。电动牵引车 LatController 不参与当前项目，本文档不复现其信号流与算法实现细节。

## 4. 纵向控制器（LonController）

### 4.1 接口

**输入**

| 变量名        | 来源                                       | 单位 | 说明                                |
|---------------|--------------------------------------------|------|-------------------------------------|
| x, y          | GlobalPose.position_enu                    | m    | 车辆 ENU 坐标                       |
| yaw_deg       | GlobalPose.euler_angles.z                  | 度→rad | 航向角                            |
| speed_kph     | CCANInfoStruct.ccan221_abs_vehspdlgt       | km/h | 车速                                |
| speed_valid   | CCANInfoStruct.ccan221_abs_vehspdlgtstatus | -    | 0=有效                              |
| accel_mps2    | CCANInfoStruct.ccan242_esp_algt            | m/s² | 纵向加速度（加速度计）              |
| gear_fb       | CCANInfoStruct.ccan123_vcu_displaygear     | -    | 档位反馈（1-9=D，11=R，其余=N/P）   |
| brake_active  | CCANInfoStruct.ccan267_ehb_brakeactive     | -    | 制动踏板状态                        |
| curvature_far | cmd.path_curvature_far                     | 1/m  | 横向控制器已写入                    |
| trajectory    | ADCTrajectory                              | -    | 规划轨迹                            |
| ctrl_enable   | bool                                       | -    | 控制使能                            |

**输出**

| 变量名        | 写入字段           | 单位   | 条件                   |
|---------------|--------------------|--------|------------------------|
| target_torque | cmd.target_torque  | Nm     | acc_out > 0 且 D 档    |
| brake         | cmd.brake          | m/s²（负值） | acc_out ≤ 0      |
| gear_location | cmd.gear_location  | GEAR_NEUTRAL/DRIVE/PARK | 始终输出 |

### 4.2 信号流

```
车辆位置/航向/速度 ─┐
规划轨迹 ──────────┼─>  Frenet 变换    ─> s_match      沿轨迹弧长
                   │   + 轨迹查询         s_dot        沿轨迹速度
                   │   (Step 1)           station_error 站位误差(m)
                   │                      = 参考弧长 - 实际弧长
                   │                      speed_error   速度误差(m/s)
                   │                      preview_accel_ref 参考加速度
                   │
                   │
                   ├─>  站位误差保护   ─> station_fnl 保护后站位误差
                   │    (Step 2)           低速（≤10km/h）时限制误差
                   │                       防止静止时误差累积导致起步冲击
                   │
站位PID ──────────┐
(Kp=0.25, Ki=0) ──┼─>  站位 PID     ─> speed_offset 速度补偿(m/s)
纯比例，无积分 ───┘   (Step 3)         位置偏远→加速，偏近→减速
                                        （外环：位置→速度设定）

低速PID(Kp=0.35) ┐
高速PID(Kp=0.34) ┼─>  速度 PID     ─> acc_closeloop 闭环加速度(m/s²)
切换阈值3.0 m/s ─┤    (Step 4)         速度偏差→加速度指令
输入限幅 ±5 m/s ─┘    含积分器          （内环：速度→加速度）

参考加速度（前馈）┐
前馈权重=1.0 ────┼─>  前馈叠加    ─> acc_cmd 总加速度指令
                 │    (Step 5)        = 闭环修正 + 轨迹参考加速度
                 │                    前馈提供基准，PID修正偏差

L1(车速→加速度上限) ┐
L2(车速→加速度下限) ┼─>  最终加速度限制 ─> acc_limited 限幅后加速度
L3(上轨acc+上升率) ─┤    (Step 6)            L1/L2 限制绝对范围
L4(上轨acc+下降率) ─┤    ☰幅值截幅            L3/L4/L5 限制变化速度
L5(车速→速率增幅) ─┘    ☰速率限幅
远预瞄曲率 ─────────┐    ─曲率保护          弯道(κ<-0.0075)收紧上下限
                    │                        极低速限制驱动加速度
                    │
IIR 低通 α=0.15 ──┐
                  ├─>  低通滤波     ─> acc_out 最终加速度(m/s²)
                  │                    平滑输出，减少传感器振荡

档位反馈 gear_fb ─┐
车速 ─────────────┼─>  档位状态机   ─> gear_req 档位请求(N/D/P)
制动状态 ─────────┘    (Step 7)          N→D：连续50帧(1s)确认
                                          D→N：连续50帧(1s)确认
                                          N→P：连续200帧(4s)确认

最终加速度 acc_out ─┐
                    │
实际加速度（传感器）┤─>  扭距计算    ─> torque_cmd 驱动扭矩(Nm)
风阻系数 coef_cd ───┤    (Step 8)         动力学模型：
滚阻系数 coef_roll ─┤    仅档位acc>0      风阻+坡阻+惯性力
整量修正 coef_delta ┤                     +加速度期望控制修正
车辆质量 2440kg ────┤                     力 × 轮径 / (效率×传动比)
传动比/效率/轮径 ───┘

                    ┌─>  输出分发     ─> acc>0: 输出(扭矩，0，档位)
                    │    (Step 9)         acc≤0: 输出(0, 制动，档位)

--- 信号流小结 ---

* **双级 PID**（Steps 3-4）：外环（站位 PID）将位置偏差转换为速度补偿量，叠加到内环（速度 PID）的设定值上。内环将速度误差转换为加速度指令。这种嵌套结构将复杂的"位置跟踪"分解为两个简单的单变量控制问题。
* **前馈补偿**（Step 5）：直接使用轨迹提供的参考加速度作为加速度命令的叠加值，一旦依赖加速度多少就生成多少命令，PID 只负责修正实际与理想的偏差。前馈使系统响应更快，超调更小。
* **双重限幅**（Step 6）：先幅值限制（L1/L2，该车速无高速时不允许猛加猛减），再速率限幅（L3/L4，防止加速度骤变）。另外根据曲率和 jerk/加速度变化率，急弯（远预瞄曲率<-0.0075）时限幅收窄至 25%/40%，保障驾驶安全。
* **扭距模型**（Step 8）：即期望加速度转换为发动机需要输出的扭矩，考虑四种阻力：风阻（与速度²成正比）、滚动阻力（与车重成正比）、坡度阻力（根据实际速度参考估算坡度）、惯性力（加速所需力 = 质量×加速度×整量修正）。再加上加速度期望 P 控制修正（补偿模型误差）再乘以轮胎半径/（传动比×效率）。
* **档位状态机**（Step 7）：N（空挡）≠ D（前进挡）≠ N → P（驻车挡）的状态机，每次转换需连续确认多拍（1-4秒），防止信号抖动导致误换挡。

### 4.2 车辆固定参数（不调整）

| 参数 | 说明 |
|------|------|
| wheel_rolling_radius | 轮胎半径（m），vehicle_param |
| transmission_efficiency | 传动效率 |
| transmission_ratio_R | R 档传动比 |
| transmission_ratio_D1 | D1 档传动比（v1 文档以有单一 D 档） |
| transmission_ratio_D2 | D2 档传动比 |
| transmission_ratio_D3 | D3 档传动比 |
| transmission_ratio_D4 | D4 档传动比 |
| windward_area | 迎风面积（m²），vehicle_param |
| ve_mass | **动态**：优先读 VCU 重量信号（`ad18fe7027_vcu_vehicleweight`，1-55吨范围有效），超范围时取默认值 **9300 kg**（v1 文档仅提示 2440 kg 整备质量，需校正） |
| kair_density | 1.2041 kg/m³（空气密度，硬编码） |
| kcoef_gravity | 9.81 m/s²（重力，硬编码） |

### 4.3 可调控制参数

**PID 增益（最核心）**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| station_pid.kp | 0.25 | 站位误差→速度补偿，比例增益 |
| station_pid.ki | 0.0 | 站位积分（当前关闭） |
| station_pid.integrator_enable | false | - |
| station_pid.sat | 0.3 | 积分限幅（m/s） |
| low_speed_pid.kp | 0.35 | 低速（< switch_speed）速度 PID Kp |
| low_speed_pid.ki | 0.01 | 低速速度 PID Ki |
| low_speed_pid.sat | 0.3 | 积分限幅（m/s²） |
| high_speed_pid.kp | 0.34 | 高速速度 PID Kp |
| high_speed_pid.ki | 0.01 | 高速速度 PID Ki |
| high_speed_pid.sat | 0.3 | - |
| switch_speed | 3.0 m/s | 低/高速 PID 切换阈值 |

**前馈与预览参数**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| preview_window | 5.0 拍 +0.1 s | 站位/速度预览参考时间 |
| preview_window_for_speed_pid | 50.0 拍+1.0 s | 速度 PID 预览参考时间 |
| acc_cmd_use_preview_point_a | 5.0 m/s | 速度 PID 输入截断 |
| station_error_limit | 5.0 m | 站位误差截幅上限 |
| speed_controller_input_limit | 5.0 m/s | - |
| acc_standstill_down_rate | -0.005 m/s²/拍 | 低速（<1.5 m/s）时的加速度下降速率 |
| a_preview_point_filt_coff | **0.05** | 预瞄加速度参考的 IIR 滤波系数（v1 文档未收录） |

**扭矩模型参数（GFlags，可运行时修改）**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| FLAGS_coef_cd | 0.6 | 风阻系数 |
| FLAGS_coef_rolling | 0.013 | 滚动阻力系数 |
| FLAGS_coef_delta | 1.05 | 转向惯量修正系数（惯性力 = Cδ × m × a）|
| FLAGS_accel_to_torque_kg | 1000 N/(m/s²) | 加速度闭环 P 控制器增益 |
| FLAGS_torque_combustion_upper_limit | **1800 Nm**（control.conf 覆盖，GFlags 默认 1200）| 扭矩输出上限 |
| FLAGS_torque_combustion_lower_limit | 0 Nm | 扭矩输出下限 |

**加速度限幅查找表**

**L1: acc_up_lim_table（speed_kph → 加速度上限）**

作用：Step 6 中 `acc_clamped = clamp(acc_cmd, L2, L1)` 的上限。
速度越高，允许的最大加速度越小（1.6~1.2 m/s²），保障高速行驶舒适性和安全性。

| speed (km/h) | 0 | 10 | 20 | 30 | 40 | 50 |
|--------------|---|----|----|----|----|----|
| acc_up_lim (m/s²) | 1.6 | 1.5 | 1.5 | 1.4 | 1.3 | 1.2 |

**L2: acc_low_lim_table（speed_kph → 加速度下限）**

作用：Step 6 中加速度下限（负值=制动）。低速时制动能力受限（0 km/h 仅 -0.1 m/s²），
防止低速怠速造成的轮胎倒拖和乘员不适。高速允许更强制动（-3.5 m/s²）。

| speed (km/h) | 0 | 1 | 2 | 4 | 12 | 25 |
|--------------|---|---|---|---|----|----|
| acc_low_lim (m/s²) | -0.1 | -0.5 | -1.5 | -2.0 | -3.0 | -3.5 |

**L3: acc_up_rate_table（前一帧 acc_out → 加速度上升率/拍）**

作用：Step 6 中加速度上升率限制 `acc_up_rate = L3(prev_acc) × L5(spd)`。
限制加速度增大的速度（jerk 限制），防止驱动系统冲击。
注意表引显**前一帧加速度值**，不是速度。在制动→加速过渡区（-0.1~0.1）速率最低（0.035）。

| prev_acc (m/s²) | -0.5 | -0.25 | -0.1 | 0.1 | 0.25 | 0.5 |
|-----------------|------|-------|------|-----|------|-----|
| up_rate (m/s²/step) | 0.045 | 0.040 | 0.035 | 0.035 | 0.040 | 0.045 |

**表 L4: acc_down_rate_table（前一帧 acc_out → 加速度下降率/拍，负值）**

作用：Step 6 中加速度下降率限制。限制加速度减小（含加速→制动过渡）的速度。
中间区域（-0.1~0.2）速率最低，过渡更平缓；两端允许更快变化（-0.030）。

| prev_acc (m/s²) | -1.0 | -0.5 | -0.3 | -0.1 | 0.1 | 0.2 | 0.5 | 1.0 |
|-----------------|------|------|------|------|-----|-----|-----|-----|
| down_rate (m/s²/step) | -0.030 | -0.030 | -0.025 | -0.020 | -0.020 | -0.020 | -0.025 | -0.030 |

**表 L5: acc_rate_gain_table（speed_kph → 速率增益）**

作用：Step 6 中 `acc_up_rate = L3(prev_acc) × L5(spd)`，对上升速率做速度修正。
低速时增益 1.5（允许更快变化以让即时响应），高速时增益 1.0（正常速率）。
注：此增益仅作用于上升速率（L3），不影响下降速率（L4）。

| speed (km/h) | 0 | 10 | 20 | 30 | 50 |
|--------------|---|----|----|----|----|
| gain | 1.5 | 1.5 | 1.35 | 1.2 | 1.0 |

### 4.4 内部状态（帧间持久）

```python
station_pid       = PID(kp=0.25, ki=0.0, ...)    # 站位 PID
speed_pid         = PID(kp=0.35, ki=0.01, ...)   # 速度 PID（参数由高/低速切换）
acc_out_prev      = 0.0                          # 上一帧最终输出加速度（CalFinalAccCmd 输出）
# 双通道低通滤波器（v1 文档误记为单个统一滤波器）：
lpf_torque        = IIR(alpha=0.15)              # 扭矩通道低通（加速时激活，制动时 Reset）
lpf_brake         = IIR(alpha=0.15)              # 制动通道低通（制动时激活，加速时 Reset）
# 预览参考滤波状态：
preview_v_filt_prev       = 0.0                  # 预览速度 IIR 状态（α=0.1）
preview_accel_ref_prev    = 0.0                  # 预览加速度 IIR 状态（α=0.05）
station_error_fnl_prev    = 0.0                  # 上一帧最终站位误差
counter_n2d       = Counter(50)                  # N→D 计数器（50拍=1s）
counter_d2n       = Counter(50)                  # D→N 计数器
counter_n2p       = Counter(200)                 # N→P 计数器（200拍=4s）
counter_p2n       = Counter(50)                  # P→N 计数器（新增，v1 文档无此状态）
```

首次使能时：重置 station_pid, speed_pid, iir_acc_state,

### 4.5 算法实现

```python
def lon_controller(x, y, yaw_deg, speed_kph, speed_valid, accel_mps2,
                   gear_fb, brake_active, curvature_far,
                   trajectory, ctrl_enable, ctrl_first_active, dt=0.02):
    speed_mps = speed_kph / 3.6
    yaw_rad   = yaw_deg * deg2rad

    if ctrl_first_active:
        station_pid.reset()
        speed_pid.reset()
        iir_acc_state = 0.0

    # Step 1: 计算误差（Frenet 坐标变换）
    matched = query_nearest_by_position(trajectory, x, y)        # PathPoint
    s_match, s_dot, d, d_dot = to_frenet(trajectory, x, y, yaw_rad,
                                         speed_mps, matched)

    t_now   = trajectory.header.measurement_time
    ref_pt  = query_nearest_by_absolute_time(trajectory, t_now)
    prev_pt = query_nearest_by_absolute_time(trajectory,
                t_now + preview_window * dt)                     # +0.1s
    spd_pt  = query_nearest_by_absolute_time(trajectory,
                t_now + preview_window_for_speed_pid * dt)       # +1.0s

    station_error = ref_pt.s - s_match
    speed_error   = ref_pt.v - s_dot

    # 预览速度参考：低速截断 + IIR 滤波（v1 文档未提及）
    if vehicle_speed_kph <= 0.5 and spd_pt.v <= 0.3:
        preview_v_cut = 0.0
    else:
        preview_v_cut = spd_pt.v
    preview_v_filt = preview_v_cut * 0.1 + preview_v_filt_prev * 0.9  # IIR α=0.1
    preview_v_filt_prev = preview_v_filt
    preview_speed_error = preview_v_filt - s_dot

    # 预览加速度参考：IIR 滤波，系数 a_preview_point_filt_coff=0.05（v1 文档未提及）
    preview_accel_ref = prev_pt.a * a_preview_point_filt_coff + preview_accel_ref_prev * (1 -
      a_preview_point_filt_coff)
    preview_accel_ref_prev = preview_accel_ref

    # Step 2: 站位误差保护
    station_limited = clamp(station_error, -station_error_limit, station_error_limit)
    # 低速特殊处理（防止静止时因误差过大导致冲出）
    # 注意：速度阈值为 1.0 km/h（v1 文档误记为 10 km/h）
    if vehicle_speed_kph > 1.0:
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

    # Step 3: 站位 PID → 速度补偿
    speed_offset = station_pid.control(station_fnl, dt)

    # Step 4: 速度 PID 切换 + 控制
    if speed_mps <= switch_speed:            # < 3.0 m/s
        speed_pid.set_pid(low_speed_pid.kp, low_speed_pid.ki, 0)
    else:
        speed_pid.set_pid(high_speed_pid.kp, high_speed_pid.ki, 0)

    speed_input = clamp(speed_offset + preview_speed_error,
                        -speed_controller_input_limit,
                        speed_controller_input_limit)
    acc_closeloop = speed_pid.control(speed_input, dt)

    # Step 5: 前馈叠加
    # 纵向前馈由三部分组成：
    #   ① 速度 PID 闭环输出
    #   ② 轨迹参考加速度前馈（经 IIR 滤波，系数 a_preview_point_filt_coff=0.05）
    #   ③ VCU 坡度补偿前馈（FLAGS_enable_slope_offset 开关，当前配置 = false，不生效）
    # 预览加速度滤波（v1 文档未提及）
    preview_accel_ref = (preview_pt.a * a_preview_point_filt_coff
                        + preview_accel_ref_prev * (1 - a_preview_point_filt_coff))
    preview_accel_ref_prev = preview_accel_ref      # a_preview_point_filt_coff = 0.05

    # VCU 坡度补偿（当前 FLAGS_enable_slope_offset=false，slope_offset=0）：
    # grade_percent = VCU.ad18ff0d27_vca_12_1bridgeleftangle()
    # slope_theta   = atan(grade_percent / 100.0)，限幅 ±20°
    # slope_offset  = digital_filter(g * sin(slope_theta))   ← 低通截止频率 5 Hz
    slope_offset = 0.0      # 当前配置下等效为 0

    acc_cmd = acc_closeloop + acc_cmd_use_preview_point_a * preview_accel_ref + slope_offset

    # Step 6: CalFinalAccCmd — 幅值 + 速率双重限制 ──
    if ctrl_enable:
        acc_up_lim       = lookup1d(L1, abs(speed_kph))
        acc_low_lim      = lookup1d(L2, abs(speed_kph))
        acc_up_rate_raw  = lookup1d(L3, acc_out_prev)
        acc_dn_rate_raw  = lookup1d(L4, acc_out_prev)
        rate_gain        = lookup1d(L5, abs(speed_kph))
        acc_up_rate      = acc_up_rate_raw * rate_gain

        # 急弯收紧（curvature_far < -0.0075 时）
        if curvature_far < -0.0075:
            acc_up_lim  *= 0.75
            acc_low_lim *= 0.60

        # 低速保护：下降率使用静止值
        if abs(speed_mps) < 1.5:
            acc_dn_rate = acc_standstill_down_rate       # = -0.005
        else:
            acc_dn_rate = acc_dn_rate_raw

        # 幅值限制
        acc_clamped = clamp(acc_cmd, acc_low_lim, acc_up_lim)

        # 低速防滑下降保护
        # 若 abs(speed_mps) < 0.04 m/s² 且 acc_clamped 在 [-0.05, 0.06] 之间：
        if abs(speed_mps) < 0.04 and -0.05 <= acc_clamped <= 0.06:
            acc_lowpass = 0.06
        else:
            acc_lowpass = acc_clamped

        acc_limited = clamp(acc_lowpass,
                            acc_out_prev + acc_dn_rate,
                            acc_out_prev + acc_up_rate)
    else:
        acc_limited = 0
        acc_out_prev = 0

    # IIR 低通（v1 文档无）：y = α·x + (1-α)·y_prev
    acc_out = acc_limited * 0.15 + iir_acc_state * (1 - 0.15)
    iir_acc_state = acc_out

    # ── Step 7: 档位状态机（GearControl）──
    # 档位编码（v1 文档缺失），当前值来自 ZJ20
    # N=0, P=1, D=2~5 (D2/D3/D4/D5), R=-1
    is_D = (2 <= gear_fb <= 5)
    is_R = (gear_fb == 1)
    is_N = (gear_fb == 0)
    current_gear_enum = 'D' if is_D else 'R' if is_R else 'N'

    # N→D: acc_cmd > 0.06（v1 文档误记为 0.01）且控制使能
    n2d_condition = (acc_cmd > 0.06 and ctrl_enable)
    counter_n2d.update(n2d_condition)
    if is_N and ctrl_enable and abs(speed_mps) <= 0.25 and counter_n2d.is_triggered():
        gear_req = 'D'
        counter_n2d.reset(); counter_d2n.reset()

    # D→N: 速度阈值 0.1 m/s（v1 文档误记为 0.01 m/s）
    d2n_condition = (acc_cmd < 0.05 and abs(speed_mps) <= 0.1 and ctrl_enable)
    counter_d2n.update(d2n_condition)
    if counter_d2n.is_triggered():
        gear_req = 'N'
        counter_d2n.reset(); counter_n2d.reset()

    # N→P: 不再检查 brake_active，改为检查 EPB 驻车状态
    n2p_condition = (is_N and ctrl_enable and acc_cmd < 0.05 and abs(speed_mps) <= 0.01)
    counter_n2p.update(n2p_condition)
    if not n2p_condition:
        counter_n2p.reset()
    if counter_n2p.is_triggered():
        gear_req = 'P'
        # 确认 EPB 状态为驻车激活（adcff9e50_epb_parkbrkst == 1）后才重置计数器

    # gear_req 默认跟随当前档位（如无上述触发）

    # ── Step 8: CalFinalTorque — 扭矩模型（D 档且 acc_cmd > -0.05）──
    # 死区阈值：kAccelerationDeadZone = -0.05 m/s²（v1 文档误记为 0）
    torque_out = 0.0
    if is_D:
        # 坡度估算（FLAGS_enable_slope_estimate 开关；当前 = false → slope=0）
            estimated_slope = estimate_slope()  # 当前配置下返回 0.0
            # 详见 §4.6 坡度估算说明

            F_air     = 0.5 * FLAGS_coef_cd * kair_density * windward_area * speed_mps**2
            F_rolling = FLAGS_coef_rolling * veh_mass * kcoef_gravity * cos(estimated_slope)
            F_slope   = veh_mass * kcoef_gravity * sin(estimated_slope)
            F_inertia = FLAGS_accel_coef_delta * veh_mass * acc_out
            F_resist  = F_air + F_rolling + F_slope + F_inertia

            error_ax = acc_out - accel_mps2
            F_P = FLAGS_accel_to_torque_kp * error_ax
            # 注：积分项 FLAGS_accel_to_torque_ki 在代码中被累积但未加入输出，可忽略

            # 多档传动比（v1 文档仅有单一 D 档比）
            # gear_fb: 1=R, 2=D1, 3=D2, 4=D3, 5=D4
            trans_ratio = {1: transmission_ratio_R,
                           2: transmission_ratio_D1, 3: transmission_ratio_D2,
                           4: transmission_ratio_D3, 5: transmission_ratio_D4}.get(gear_fb, 1)

            T_raw = (F_resist + F_P) * wheel_rolling_radius / (transmission_efficiency * trans_ratio)
            torque_clamped = clamp(T_raw,
                                   FLAGS_torque_combustion_lower_limit,
                                   FLAGS_torque_combustion_upper_limit)

            if acc_out > kAccelerationDeadZone:   # > -0.05
                torque_out = torque_clamped

    # --- Step 9: 输出分发 + 双通道滤波 ---
    # v1 文档：一个 IIR(α=0.15) 作用于 acc_limited 后再分支
    # 当前代码：两个独立滤波器分别作用于电机通道和制动通道，互斥 Reset

    # 死区判断（kAccelerationDeadZone = -0.05 m/s²）
    if acc_out > kAccelerationDeadZone:   # 扭矩模式：[-0.05, +∞)
        # 扭矩低通（α=0.15）
        torque_filtered = lpf_torque.update(torque_out)   # α=0.15，同 LowPassfilter_Torque
        lpf_brake.reset()                                  # 制动滤波器 Reset
        return torque_filtered, 0.0, gear_req
    else:                                  # 制动模式：(-∞, -0.05)
        # 制动低通（α=0.15）
        brake_filtered = lpf_brake.update(acc_out)        # α=0.15，同 LowPassfilter_1
        lpf_torque.reset()                                 # 扭矩滤波器 Reset
        return 0.0, brake_filtered, gear_req
```


### 4.6 坡度估计算法（EstimateSlope）

**启用开关**：`FLAGS_enable_slope_estimate`（当前默认 `false`，坡度估计禁用，`slope_ = 0`）

当启用时，算法通过 IMU 加速度与车轮加速度，估计道路坡度角。

#### 4.6.1 信号流

```
CAN车速 raw_speed ─────┐
                       │  ┌─────────────┐
                       ├─▶│ 车速预处理   │──▶ ego_speed_mps 滤波后车速
                       │  │ + 加速度计算 │    ego_accel_wheel 车轮加速度
                       │  │ (Step 1)    │   （速度差分 / 周期）
                       │  └─────────────┘
                       │
IMU加速度 raw_accel_imu ─▶ ┌─────────────┐
                           │ IMU预处理    │──▶ ego_accel_imu 滤波后IMU加速度
                           │ (Step 2)    │
                           └─────────────┘

横摆角速度 raw_yawrate ──▶ ┌─────────────┐
                           │ 信号历史队列 │──▶ 三组 25帧历史数据
                           │ (Step 3)    │     accel_wheel_history[]
                           └─────────────┘     accel_imu_history[]
                                               yawrate_history[]

历史数据 ────────────────▶ ┌─────────────┐
                           │ 加权平均滤波 │──▶ ego_accel_wheel（加权平均）
                           │ (Step 4)    │    ego_accel_imu（加权平均）
                           └─────────────┘    ego_yawrate_rads（加权平均）

静止状态 ────────────────▶ ┌─────────────┐
                           │ 零偏估计     │──▶ accel_imu_bias 加速度零偏
                           │ (Step 5)    │    yawrate_bias 横摆零偏
                           └─────────────┘   （仅在速度<0.01m/s时累积）

车速/加速度历史 ──────────▶ ┌─────────────┐
                           │ 特殊工况识别 │──▶ isSpecialScene 标志
                           │ (Step 6)    │   （急加速/刹停至静止/起步/静止）
                           └─────────────┘

横向加速度 est_ay ────────▶ ┌─────────────┐
                            │ 弯道保持策略 │──▶ 若 est_ay > 2.5 m/s²
                            │ (Step 7)    │    保持前一帧 road_slope
                            └─────────────┘   （考虑 4 种超限情况）

ego_accel_imu ────────────▶ ┌─────────────┐
ego_accel_wheel ──────────▶ │ 瞬时计算     │──▶ g_sin_theta = IMU减轮加速度
                            │ (Step 8)    │    + IMU航向投影分量
                            └─────────────┘    - 车轮加速度动态项
                                               - raw_slope = asin(g_sin_theta/g)

previous_raw_slope ───────▶ ┌─────────────┐
                            │ 防跳变检测   │──▶ 变化率限制 1°/s
                            │ (Step 9)    │
                            └─────────────┘

isSpecialScene ───────────▶ ┌─────────────┐
previous_road_slope ──────▶ │ 滤波更新     │──▶ road_slope 滤波后坡度(rad)
                            │ (Step 10)   │    正常：α=0.07
                            └─────────────┘    特殊：α=0.0185

                            ┌─────────────┐
                            │ 饱和+限幅    │──▶ road_slope 最终输出(rad)
                            │             │    |slope| ≤ 15°
                            └─────────────┘
                            输出 road_slope
                            范围 ±15°
```


#### 4.6.2 算法参数

**硬编码常量**

| 常量                       | 值        | 说明                         |
|----------------------------|-----------|------------------------------|
| kSize                      | 25        | 历史数据队列长度（帧数）      |
| kSpeedFilterCoef           | 0.3       | 车速滤波系数                  |
| kImuAccFilterCoef          | 0.2       | IMU加速度滤波系数             |
| kSlopeFilterCoefOrdinary   | 0.07      | 普通工况坡度滤波系数          |
| kSlopeFilterCoefSpecial    | 0.0105    | 特殊工况坡度滤波系数（更保守）|
| kSlopeRateLimit            | 5° (rad)  | 坡度变化率上限                |
| kSlopeLimitRad             | 3° (rad)  | 坡度幅值上限                  |
| kSlopeDeadZone             | 1° (rad)  | 坡度死区                      |
| kSlopeHoldThreshold        | 2.5 m/s²  | 弯道保持策略阈值              |
| accel_timewindow_          | 500 (帧)  | 加速度零偏估计窗口（10秒）    |
| yawrate_timewindow_        | 500 (帧)  | 横摆率零偏估计窗口（10秒）    |

**加权滤波系数组（slope_filter_coef[25]）**

用于 Step 4 的加权平均滤波，系数递增形成线性加权窗：

```python
slope_filter_coef = [
    0.003076923076923, 0.006153846153846, 0.009230769230769, 0.012307692307692,
    0.015384615384615, 0.018461538461538, 0.021538461538462, 0.024615384615385,
    0.027692307692308, 0.030769230769231, 0.033846153846154, 0.036923076923077,
    0.040000000000000, 0.043076923076923, 0.046153846153846, 0.049230769230769,
    0.052307692307692, 0.055384615384615, 0.058461538461538, 0.061538461538462,
    0.064615384615385, 0.067692307692308, 0.070769230769231, 0.073846153846154,
    0.076923076923077
]  # 总和 = 1.0
```

含义：最新帧权重最高（0.077），25帧前权重最低（0.003），形成渐进遗忘窗。

**IMU安装参数（vehicle_param）**

| 参数                    | 说明                                    |
|-------------------------|-----------------------------------------|
| yawsensor_imu_L         | IMU yaw传感器距车辆中心的距离（m）      |
| yawsensor_imu_theta     | IMU yaw传感器的安装角度（rad）          |

这些参数用于补偿 IMU 位于非车辆中心时的运动学误差。


#### 4.6.3 内部状态

```python
# 信号历史队列
ego_accel_wheel_history     = [0.0] * 25   # 车轮加速度历史(m/s²)
ego_accel_imu_history       = [0.0] * 25   # IMU加速度历史(m/s²)
ego_yawrate_history         = [0.0] * 25   # 横摆角速度历史(rad/s)

# 零偏估计器
accel_bias_estimator = BiasEstimator(window_size=500)   # 10秒窗口
yawrate_bias_estimator = BiasEstimator(window_size=500)

# 滤波器前一帧状态
previous_raw_ego_speed_mps       = 0.0   # 前一帧原始车速
previous_ego_speed_mps           = 0.0   # 前一帧滤波后车速
previous_raw_ego_accel_imu       = 0.0   # 前一帧原始IMU加速度
previous_raw_slope_              = 0.0   # 前一帧原始坡度（变化率限制前）
previous_road_slope_             = 0.0   # 前一帧滤波后坡度

# 特殊工况识别用历史
ego_speed_mps_history_10         = [0.0] * 10   # 车速10帧历史
ego_accel_wheel_history_5        = [0.0] * 5    # 车轮加速度5帧历史

# 零偏
accel_imu_bias = 0.0
yawrate_bias   = 0.0

timer = 0       # 帧计数器
```

**零偏估计器（BiasEstimator）实现**

```python
class BiasEstimator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.window_data = []
        self.current_bias = 0.0

    def add_data(self, data: float) -> None:
        self.window_data.append(data)
        if len(self.window_data) > self.window_size:
            self.window_data.pop(0)
        self.current_bias = sum(self.window_data) / len(self.window_data)

    def get_bias(self) -> float:
        return self.current_bias

    def get_data_count(self) -> int:
        return len(self.window_data)

    def clear(self) -> None:
        self.window_data.clear()
        self.current_bias = 0.0
```


#### 4.6.4 算法实现

```python
def estimate_slope(raw_speed_kph, raw_accel_imu, raw_yawrate, steerwheel_angle_deg,
                   vehicle_param, control_period, FLAGS_enable_slope_estimate,
                   dt=0.02):
    """瞬时坡度估算（当 FLAGS_enable_slope_estimate=false 时返回 0）"""
    if not FLAGS_enable_slope_estimate:
        return 0.0

    # 常量
    kGravitationalAcc = 9.81
    kDeg2Rad = π / 180   # 0.017453

    # --- Step 1: 车速预处理 + 车轮加速度计算 ---
    raw_speed_mps = raw_speed_kph / 3.6

    # 生成滤波后车速
    delta_speed = clamp(raw_speed_mps - previous_raw_ego_speed_mps, -0.18, 0.1)
    raw_speed_mps = previous_raw_ego_speed_mps + delta_speed
    previous_raw_ego_speed_mps = raw_speed_mps

# 车速滤波 ( IIR α=0.3)
ego_speed_mps = 0.3 * raw_speed_mps + 0.7 * previous_ego_speed_mps
previous_ego_speed_mps = ego_speed_mps

# 车轮加速度 = 速度差分 / 周期
ego_accel_wheel = (ego_speed_mps - previous_ego_speed_mps) / control_period

# 更新历史队列
ego_accel_wheel_history.pop()
ego_accel_wheel_history.insert(0, ego_accel_wheel)

# - Step 2: IMU加速度预处理
# IIR 滤波 (α=0.2)
ego_accel_imu = 0.2 * raw_accel_imu + 0.8 * previous_raw_ego_accel_imu
previous_raw_ego_accel_imu = ego_accel_imu

ego_accel_imu_history.pop()
ego_accel_imu_history.insert(0, ego_accel_imu)

# - Step 3: 横摆角速度历史
ego_yawrate_history.pop()
ego_yawrate_history.insert(0, raw_yawrate)

# - Step 4: 计数器 + 加权平均滤波
timer += 1
if timer > 25:
    timer = 25

# 加权平均 ( 使用 slope_filter_coef )
ego_accel_wheel_avg = sum(slope_filter_coef[i] * ego_accel_wheel_history[i]
                          for i in range(25))
ego_accel_imu_avg   = sum(slope_filter_coef[i] * ego_accel_imu_history[i]
                          for i in range(25))
ego_yawrate_avg     = sum(slope_filter_coef[i] * ego_yawrate_history[i]
                          for i in range(25))

# - Step 5: 零偏估计 ( 仅静止时累积 )
if abs(ego_speed_mps) < 0.01:
    accel_bias_estimator.add_data(ego_accel_imu_avg)
    yawrate_bias_estimator.add_data(ego_yawrate_avg)

    if accel_bias_estimator.get_data_count() >= 500:
        accel_imu_bias = accel_bias_estimator.get_bias()
    if yawrate_bias_estimator.get_data_count() >= 500:
        yawrate_bias = yawrate_bias_estimator.get_bias()

# 减去零偏
ego_accel_imu_avg -= accel_imu_bias
ego_yawrate_avg   -= yawrate_bias

# - Step 6: 特殊工况识别
# 更新车速10帧历史
ego_speed_mps_history_10.pop()
ego_speed_mps_history_10.insert(0, ego_speed_mps)
sum_speed_10 = sum(ego_speed_mps_history_10)

# 更新加速度5帧历史
ego_accel_wheel_history_5.pop()
ego_accel_wheel_history_5.insert(0, ego_accel_wheel_avg)
sum_accel_5 = sum(ego_accel_wheel_history_5)

# 急加速: 总速度>1.0m 且 总加速度>1.5m/s²
is_rapid_acceleration = (sum_speed_10 > 1.0) and (abs(sum_accel_5) > 1.5)

# 制动至静止: 总速度<5.0m 且 总加速度<-1.5m/s²
is_brake_to_standstill = (sum_speed_10 < 5.0) and (sum_accel_5 < -1.5)

# 起步: 总速度<1.0m 且 总加速度>0.5m/s²
is_standstill_to_move = (sum_speed_10 < 1.0) and (sum_accel_5 > 0.5)

# 静止: 总速度<0.5m 且 总加速度<0.2m/s² 且是上次状态
is_standstill = (sum_speed_10 < 0.5) and (sum_accel_5 < 0.2)

is_special_scene = (is_rapid_acceleration or is_brake_to_standstill
                    or is_standstill_to_move or is_standstill)

# - Step 7: 弯道保持策略
# 若横向加速度过大 ( 弯道 ), 保持上一帧坡度不更新
est_ay = abs(ego_speed_mps * ego_yawrate_avg)    # 横向加速度 = v × ω
if est_ay > 2.5:
    return previous_road_slope_    # 弯道中不更新

# - Step 8: 坡度计算 ( 需 timer >= 25 )
if timer < 25:
    return 0.0

# 方法: 港口方案 ( 考虑IMU安装位置 )
# g_sin_theta = IMU加速度 + IMU位置修正项 - 车轮加速度×速度修正
L_imu = vehicle_param.yawsensor_imu_L
theta_imu = vehicle_param.yawsensor_imu_theta

g_sin_theta = (ego_accel_imu_avg
               + L_imu * cos(theta_imu) * ego_yawrate_avg**2    # IMU离心加速度修正
               - ego_accel_wheel_avg * (ego_speed_mps
                           + L_imu * sin(theta_imu) * ego_yawrate_avg)
                           / max(0.1, ego_speed_mps))

g_sin_theta = clamp(g_sin_theta, -2.0, 2.0)    # 物理限制
raw_slope = asin(g_sin_theta / kGravitationalAcc)    # rad

# - Step 9: 变化率限制
delta_slope = clamp(raw_slope - previous_raw_slope_,
                    -5 * kDeg2Rad * control_period,
                    5 * kDeg2Rad * control_period)

raw_slope = previous_raw_slope_ + delta_slope
previous_raw_slope_ = raw_slope

# - Step 10: 双模式滤波
if is_special_scene:
    # 特殊工况: 更保守滤波 ( α=0.0105 )
    road_slope = 0.0105 * raw_slope + 0.9895 * previous_road_slope_
else:
    # 普通工况: 正常滤波 ( α=0.07 )
    road_slope = 0.07 * raw_slope + 0.93 * previous_road_slope_

previous_road_slope_ = road_slope

# - Step 11: 死区+限幅
if abs(road_slope) < 1 * kDeg2Rad:
    road_slope = 0.0    # 死区

road_slope = clamp(road_slope, -3 * kDeg2Rad, 3 * kDeg2Rad)    # 幅值限制

return road_slope
```

#### 4.6.5 坡度估计原理说明

**核心思想**: 对比两种加速度测量来源的偏差来估计坡度。

| 测量来源 | 物理含义 | 坡度影响 |
|---------|---------|---------|
| IMU加速度计 | 测量车辆实际加速度 + 重力分量 | 包含 `g×sin(θ)` ( 坡度分量 ) |
| 车轮加速度计 | 测量车辆动力学加速度 | 不含重力分量 |

**推导**:

```
IMU加速度 = 车轮加速度 + g×sin(坡度角)
→ g×sin(θ) = IMU加速度 - 车轮加速度
→ θ = asin((IMU加速度 - 车轮加速度) / g)
```

**IMU安装位置修正**: 当 IMU 不位于车辆中心时, 横摆运动会产生额外的离心加速度:

```
修正项 = L_imu × cos(θ_imu) × ω²          (离心加速度)
       - 车轮加速度 × L_imu × sin(θ_imu) × ω / v    (速度修正)
```

**特殊工况滤波策略**: 急加速、制动、起步、静止时采用更保守的滤波系数 ( α=0.0105 ), 防止瞬态工况导致的坡度估计跳变。

**弯道保持策略**: 当估计横向加速度 > 2.5 m/s² 时 ( 弯道 ), 不更新坡度, 保持上一帧值。原因: 弯道中横摆角速度大, IMU修正项误差增大。

#### 4.6.6 启用条件

当前配置 `FLAGS_enable_slope_estimate = false`, 坡度估计功能禁用。若需启用:

1. 设置 `FLAGS_enable_slope_estimate = true`
2. 确保 VCU 提供 IMU 安装参数 ( `yawsensor_imu_L`、`yawsensor_imu_theta` )
3. 验证零偏估计器在静止工况下正确累积

**注意**: 启用坡度估计后, `FLAGS_enable_slope_offset` 仍独立控制坡度前馈补偿是否叠加入 `acc_cmd` ( Step 5 ), 两者需同时启用才能实现完整坡度补偿。


## 5. 参数快速索引

### 5.1 固定车辆参数 ( 两个控制器共用, 不参与调参 )

| 参数名 | 来源 | 含义 |
|-------|-----|------|
| wheelbase | vehicle_param.wheel_base | 轴距 ( m ) |
| steer_ratio | vehicle_param.steer_ratio | 转向比 |
| wheel_rolling_radius | vehicle_param.wheel_rolling_radius | 轮胎半径 ( m ) |
| transmission_efficiency | vehicle_param.transmission_efficiency | 传动效率 |
| transmission_ratio_D/R | vehicle_param | 档位传动比 |
| windward_area | vehicle_param.windward_area | 迎风面积 ( m² ) |

### 5.2 可微调控制器参数汇总

**横向 ( 重卡 )**: 表 T1-T8 的各节点 value ( 共 8 张表, 全部以 speed_kph 为索引 )

| 表 | 信号流角色 | 当前状态 |
|----|-----------|---------|
| T1 | 横向偏移-转向的幅值上限 | 全速 3.86° (统一) |
| T2 | 预瞄距离-比例增益 ( 类 Kp ) | 全速 1.5s ( 统一 ) |
| T3 | 收敛时间-闭环带宽 ( 类 Kp ) | 全速 1.1s ( 统一 ) |
| T4 | 角度双阶段权重 | 全速 0-10 km/h 关闭, 之后 0.3s |
| T5 | 近预瞄前馈偏移 ( 不直接参与反馈控制环 ) | 全速 0.1s |
| T6 | 远预瞄前馈偏移 ( 曲率偏移 ) | 全速 1.0s |
| T7 | 输出坐标偏移上限 | 低速 1100°, 230 km/h 500° |
| T8 | 侧偏修正增益 | 全速 1.0 ( 未启用 ) |

**纵向**:

- 标量: station_kp, low_kp/ki, high_kp/ki, switch_speed, acc_cmd_use_preview_point_a, a_preview_point_filt_coff,
  FLAGS_accel_to_torque_kp, FLAGS_coef_cd/rolling/delta

| 表 | 信号流角色 | 当前状态 |
|----|-----------|---------|
| L1 | 加速度上限 ( 速度相关 ) | 1.6→1.2 m/s² ( 速度越高越低 ) |
| L2 | 加速度下限/最大制动 ( 速度相关 ) | -1.5→-3.5 m/s² ( 速度限制制动 ) |
| L3 | 加速度上升速率 ( jerk上限 ) | 0.035-0.045 m/s²/step |
| L4 | 加速度下降速率 ( jerk下限 ) | -0.020→-0.030 m/s²/step |
| L5 | 上升速率的速度增益 | 低速 1.5→高速 1.0 |

**纵向新增功能 ( v2 补丁 )**:

- `a_preview_point_filt_coff=0.05` : 预览加速度考虑滤波系数, 参与 Step 5
- 预览速度参考: 低速截断 ( <0.5 km/h 或 <0.3 m/s 时截断 ), IIR α=0.1, 参与 Step 4
- VCU 坡度前馈 ( `FLAGS_enable_slope_offset`, 当前=false 不生效 )
- 坡度估计 ( `FLAGS_enable_slope_estimate`, 当前=false, 完整算法见 §4.6 )
- 双通道供加速度 ( 扭矩通道 / 制动通道各有 IIR α=0.15, 含有 Reset )
- 加速度死区: `kAccelerationDeadZone=0.05 m/s²` ( 非 0 )

**坡度估计参数 ( §4.6, 当前禁用 )**

| 参数 | 当前值 | 说明 |
|------|-------|------|
| FLAGS_enable_slope_estimate | false | 坡度估计启用开关 ( GFlag ) |
| FLAGS_enable_slope_offset | false | 坡度前馈补偿启用开关 ( GFlag ) |
| kSize | 25 | 历史数据队列长度 ( 帧 ) |
| kSpeedFilterCoef | 0.3 | 车速 IIR 滤波系数 |
| kImuAccFilterCoef | 0.2 | IMU加速度 IIR 滤波系数 |
| kSlopeFilterCoefOrdinary | 0.07 | 普通工况坡度滤波系数 |
| kSlopeFilterCoefSpecial | 0.0105 | 特殊工况坡度滤波系数 ( 保守 ) |
| kSlopeRateLimit | 5° | 坡度变化率上限 ( rad ) |
| kSlopeLimitRad | 3° | 坡度幅值上限 ( rad ) |
| kSlopeDeadZone | 1° | 坡度死区 ( rad ) |
| kSlopeHoldThreshold | 2.5 m/s² | 弯道保持策略横向加速度阈值 |
| accel_timewindow | 500 帧 | 加速度零偏估计窗口 ( =10秒 ) |
| yawrate_timewindow | 500 帧 | 横摆率零偏估计窗口 ( =10秒 ) |
| yawsensor_imu_L | vehicle_param | IMU yaw传感器距车辆中心距离 ( m ) |
| yawsensor_imu_theta | vehicle_param | IMU yaw传感器安装角度 ( rad ) |

### 5.3 不参与调参的量

| 类别 | 例子 | 原因 |
|------|------|------|
| 物理/传感信号 | 车速、位置、加速计、航向角 | 输入, 非设计量 |
| 车辆固定常数 | 轴距、传动比、轮胎半径 | 由硬件决定 |
| 底盘标定表 | calibration_table(speed, acc → torque_cmd) | 描述底盘非线性, 由实车标定 |
| 硬编码安全约束 | kRate_limit_fb/ff/total、带阻滤波器系数 | 稳定性/安全保障, 不参与稳态优化 |
| 档位状态计数值 | N+D 计数器阈值 50 拍 | 安全逻辑 |
