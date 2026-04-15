# sim/sim_loop.py
"""50Hz 闭环仿真主循环。参数从配置文件加载。
V2: 支持 differentiable=True/False 双路径，兼容梯度回传。
"""
import math
import torch
from common import normalize_angle, TrajectoryPoint
from config import load_config
from model.trajectory import TrajectoryAnalyzer
from model.vehicle_factory import create_vehicle
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController

DEG2RAD = math.pi / 180.0


def run_simulation(trajectory: list[TrajectoryPoint],
                   init_speed: float = 0.0,
                   init_x: float | None = None,
                   init_y: float | None = None,
                   init_yaw: float | None = None,
                   cfg: dict | None = None,
                   lat_ctrl: LatControllerTruck | None = None,
                   lon_ctrl: LonController | None = None,
                   differentiable: bool = False,
                   tbptt_k: int = 0,
                   ) -> list[dict]:
    """运行闭环仿真。返回历史记录。

    Args:
        trajectory: 参考轨迹点列表
        init_speed: 初始速度 (m/s)
        init_x/y/yaw: 初始位姿（None 时使用轨迹起点）
        cfg: 配置字典（None 时加载默认配置）
        lat_ctrl: 外部横向控制器（None 时内部创建）
        lon_ctrl: 外部纵向控制器（None 时内部创建）
        differentiable: True 时全程 tensor 运算，支持梯度回传
        tbptt_k: Truncated BPTT 窗口大小（步数）。>0 时每 K 步 detach 车辆状态，
                 截断梯度链以防止梯度爆炸。0 表示不截断（完整 BPTT）。

    Returns:
        历史记录列表。differentiable=True 时值为 tensor，False 时为 float。
    """
    if cfg is None:
        cfg = load_config()

    veh = cfg['vehicle']
    model_type = veh.get('model_type', 'kinematic')
    if model_type == 'dynamic':
        dyn = cfg['dynamic_vehicle']
        wheelbase = dyn['lf'] + dyn['lr']
        steer_ratio = dyn['steer_ratio']
    elif model_type == 'hybrid_dynamic':
        hyb = cfg['hybrid_dynamic_vehicle']
        wheelbase = hyb['lf'] + hyb['lr']
        steer_ratio = hyb['steer_ratio']
    elif model_type == 'hybrid_v2':
        params_key = cfg['vehicle'].get('params_section', 'dynamic_v2_vehicle')
        v2p = cfg[params_key]
        wheelbase = v2p['lf'] + v2p['lr']
        steer_ratio = v2p['steer_ratio']
    else:
        wheelbase = veh['wheelbase']
        steer_ratio = veh['steer_ratio']
    dt = cfg['simulation']['dt']

    analyzer = TrajectoryAnalyzer(trajectory)
    traj_duration = trajectory[-1].t

    p0 = trajectory[0]
    x0 = init_x if init_x is not None else p0.x
    y0 = init_y if init_y is not None else p0.y
    yaw0 = init_yaw if init_yaw is not None else p0.theta

    car = create_vehicle(cfg, x=x0, y=y0, yaw=yaw0, v=init_speed,
                         dt=dt, differentiable=differentiable)

    # 控制器：外部传入时重置状态，否则内部创建
    if lat_ctrl is None:
        lat_ctrl = LatControllerTruck(cfg, differentiable=differentiable)
    else:
        lat_ctrl.reset_state()

    if lon_ctrl is None:
        lon_ctrl = LonController(cfg, differentiable=differentiable)
    else:
        lon_ctrl.reset_state()

    history = []
    n_steps = int(traj_duration / dt)
    prev_steer = torch.tensor(0.0) if differentiable else 0.0

    # kinematic 走旧加速度路径，其他走新扭矩路径
    use_torque_layer = (model_type != 'kinematic')
    v_prev = init_speed  # 用于差分得到 a_actual（扭矩路径需要）

    for step in range(n_steps):
        t = step * dt

        if differentiable:
            # Truncated BPTT: 每 K 步 detach 车辆状态，截断梯度链
            if tbptt_k > 0 and step > 0 and step % tbptt_k == 0:
                car.detach_state()
                prev_steer = prev_steer.detach()

            # ── TENSOR 路径：全程保持 tensor，梯度可回传 ──
            ref_pt = analyzer.query_nearest_by_position(car.x, car.y)
            ref_theta = torch.tensor(ref_pt.theta)
            dx = car.x - ref_pt.x
            dy = car.y - ref_pt.y
            lateral_error = torch.cos(ref_theta) * dy - torch.sin(ref_theta) * dx
            heading_error = normalize_angle(car.yaw - ref_theta)

            # 估算横摆角速度
            delta_prev = prev_steer / steer_ratio * DEG2RAD
            yawrate = car.v * torch.tan(delta_prev) / wheelbase

            # 横向控制器
            steer_out, kappa_cur, kappa_near, curvature_far, steer_fb, steer_ff = \
                lat_ctrl.compute(
                    x=car.x, y=car.y,
                    yaw_deg=car.yaw_deg,
                    speed_kph=car.speed_kph,
                    yawrate=yawrate,
                    steer_feedback=prev_steer,
                    analyzer=analyzer,
                    ctrl_enable=True, dt=dt)

            # 纵向控制器
            acc_cmd = lon_ctrl.compute(
                x=car.x, y=car.y,
                yaw_deg=car.yaw_deg,
                speed_kph=car.speed_kph,
                accel_mps2=torch.tensor(0.0),
                curvature_far=curvature_far,
                analyzer=analyzer, t_now=t,
                ctrl_enable=True,
                ctrl_first_active=(step == 0), dt=dt)

            # 车辆更新：kinematic 吃 acc，动力学/混合吃车轮扭矩
            delta_front = steer_out / steer_ratio * DEG2RAD
            if use_torque_layer:
                # a_actual 从上一步速度差分，v_prev 已 detach（或首帧为常数）
                a_actual = (car.v - v_prev) / dt
                torque_wheel = lon_ctrl.compute_torque_wheel(
                    acc_cmd, car.v, a_actual)
                history.append({
                    't': t,
                    'x': car.x, 'y': car.y, 'yaw': car.yaw, 'v': car.v,
                    'steer': steer_out, 'steer_fb': steer_fb,
                    'steer_ff': steer_ff,
                    'acc': acc_cmd, 'torque_wheel': torque_wheel,
                    'lateral_error': lateral_error,
                    'heading_error': heading_error,
                    'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
                })
                v_prev = car.v.detach()
                car.step(delta=delta_front, torque_wheel=torque_wheel)
            else:
                history.append({
                    't': t,
                    'x': car.x, 'y': car.y, 'yaw': car.yaw, 'v': car.v,
                    'steer': steer_out, 'steer_fb': steer_fb,
                    'steer_ff': steer_ff,
                    'acc': acc_cmd,
                    'lateral_error': lateral_error,
                    'heading_error': heading_error,
                    'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
                })
                car.step(delta=delta_front, acc=acc_cmd)
            prev_steer = steer_out

        else:
            # ── FLOAT 路径：V1 兼容行为，.item() 提取 ──
            car_x = car.x.item()
            car_y = car.y.item()
            car_yaw = car.yaw.item()
            car_v = car.v.item()
            car_speed_kph = car.speed_kph.item()

            # 当前参考点
            ref_pt = analyzer.query_nearest_by_position(car_x, car_y)
            dx = car_x - ref_pt.x
            dy = car_y - ref_pt.y
            lateral_error = (math.cos(ref_pt.theta) * dy
                             - math.sin(ref_pt.theta) * dx)
            heading_error = normalize_angle(car_yaw - ref_pt.theta).item()

            # 估算横摆角速度
            delta_prev = prev_steer / steer_ratio * DEG2RAD
            yawrate = car_v * math.tan(delta_prev) / wheelbase

            # 横向控制器
            steer_out, kappa_cur, kappa_near, curvature_far, steer_fb, steer_ff = \
                lat_ctrl.compute(
                    x=car_x, y=car_y,
                    yaw_deg=math.degrees(car_yaw),
                    speed_kph=car_speed_kph,
                    yawrate=yawrate,
                    steer_feedback=prev_steer,
                    analyzer=analyzer,
                    ctrl_enable=True, dt=dt)

            # 纵向控制器
            acc_cmd = lon_ctrl.compute(
                x=car_x, y=car_y,
                yaw_deg=math.degrees(car_yaw),
                speed_kph=car_speed_kph,
                accel_mps2=0.0,
                curvature_far=curvature_far,
                analyzer=analyzer, t_now=t,
                ctrl_enable=True,
                ctrl_first_active=(step == 0), dt=dt)

            # 车辆更新：kinematic 吃 acc，动力学/混合吃车轮扭矩
            delta_front = steer_out / steer_ratio * DEG2RAD
            if use_torque_layer:
                a_actual = (car_v - v_prev) / dt
                torque_wheel = lon_ctrl.compute_torque_wheel(
                    acc_cmd, car_v, a_actual)
                history.append({
                    't': t, 'x': car_x, 'y': car_y, 'yaw': car_yaw,
                    'v': car_v, 'steer': steer_out, 'steer_fb': steer_fb,
                    'steer_ff': steer_ff,
                    'acc': acc_cmd, 'torque_wheel': torque_wheel,
                    'lateral_error': lateral_error,
                    'heading_error': heading_error,
                    'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
                })
                v_prev = car_v
                car.step(delta=delta_front, torque_wheel=torque_wheel)
            else:
                history.append({
                    't': t, 'x': car_x, 'y': car_y, 'yaw': car_yaw,
                    'v': car_v, 'steer': steer_out, 'steer_fb': steer_fb,
                    'steer_ff': steer_ff, 'acc': acc_cmd,
                    'lateral_error': lateral_error,
                    'heading_error': heading_error,
                    'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
                })
                car.step(delta=delta_front, acc=acc_cmd)
            prev_steer = steer_out

    return history
