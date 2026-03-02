# sim/sim_loop.py
"""50Hz 闭环仿真主循环。参数从配置文件加载。"""
import math
from common import normalize_angle, TrajectoryPoint
from config import load_config
from trajectory import TrajectoryAnalyzer
from vehicle import BicycleModel
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController

DEG2RAD = math.pi / 180.0


def run_simulation(trajectory: list[TrajectoryPoint],
                   init_speed: float = 0.0,
                   init_x: float | None = None,
                   init_y: float | None = None,
                   init_yaw: float | None = None,
                   cfg: dict | None = None,
                   ) -> list[dict]:
    """运行闭环仿真。返回历史记录。"""
    if cfg is None:
        cfg = load_config()

    veh = cfg['vehicle']
    wheelbase = veh['wheelbase']
    steer_ratio = veh['steer_ratio']
    dt = cfg['simulation']['dt']

    analyzer = TrajectoryAnalyzer(trajectory)
    traj_duration = trajectory[-1].t

    p0 = trajectory[0]
    x0 = init_x if init_x is not None else p0.x
    y0 = init_y if init_y is not None else p0.y
    yaw0 = init_yaw if init_yaw is not None else p0.theta

    car = BicycleModel(wheelbase=wheelbase, x=x0, y=y0,
                       yaw=yaw0, v=init_speed, dt=dt)
    lat_ctrl = LatControllerTruck(cfg)
    lon_ctrl = LonController(cfg)

    history = []
    n_steps = int(traj_duration / dt)
    prev_steer = 0.0

    for step in range(n_steps):
        t = step * dt

        # 当前参考点
        ref_pt = analyzer.query_nearest_by_position(car.x, car.y)
        dx = car.x - ref_pt.x
        dy = car.y - ref_pt.y
        lateral_error = (math.cos(ref_pt.theta) * dy
                         - math.sin(ref_pt.theta) * dx)
        heading_error = normalize_angle(car.yaw - ref_pt.theta)

        # 估算横摆角速度（用上一步的前轮转角）
        delta_prev = prev_steer / steer_ratio * DEG2RAD
        yawrate = car.v * math.tan(delta_prev) / wheelbase

        # 横向控制器
        steer_out, kappa_cur, kappa_near, curvature_far = lat_ctrl.compute(
            x=car.x, y=car.y,
            yaw_deg=math.degrees(car.yaw),
            speed_kph=car.speed_kph,
            yawrate=yawrate,
            steer_feedback=prev_steer,
            analyzer=analyzer,
            ctrl_enable=True, dt=dt)

        # 纵向控制器
        acc_cmd = lon_ctrl.compute(
            x=car.x, y=car.y,
            yaw_deg=math.degrees(car.yaw),
            speed_kph=car.speed_kph,
            accel_mps2=0.0,
            curvature_far=curvature_far,
            analyzer=analyzer, t_now=t,
            ctrl_enable=True,
            ctrl_first_active=(step == 0), dt=dt)

        history.append({
            't': t, 'x': car.x, 'y': car.y, 'yaw': car.yaw,
            'v': car.v, 'steer': steer_out, 'acc': acc_cmd,
            'lateral_error': lateral_error, 'heading_error': heading_error,
            'ref_x': ref_pt.x, 'ref_y': ref_pt.y,
        })

        # 车辆更新
        delta_front = steer_out / steer_ratio * DEG2RAD
        car.step(delta=delta_front, acc=acc_cmd)
        prev_steer = steer_out

    return history
