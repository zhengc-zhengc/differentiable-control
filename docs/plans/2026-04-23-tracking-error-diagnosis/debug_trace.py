"""临时调试脚本：在 truck_trailer 上跑圆弧，打印状态轨迹，定位稳态误差 / 发散源。"""
import sys
import math
sys.path.insert(0, '.')
from config import load_config
from model.trajectory import generate_circle, generate_lane_change
from sim_loop import run_simulation


def run_and_dump(name, traj, cfg, init_speed):
    print(f"=== {name} ===")
    hist = run_simulation(traj, init_speed=init_speed, cfg=cfg,
                          differentiable=False)
    print(f"steps={len(hist)}")
    print(f"ref start: x={traj[0].x:.2f} y={traj[0].y:.2f} "
          f"theta={math.degrees(traj[0].theta):.2f}deg "
          f"kappa={traj[0].kappa:.4f} v={traj[0].v:.2f}")
    print(f"ref end:   x={traj[-1].x:.2f} y={traj[-1].y:.2f} "
          f"theta={math.degrees(traj[-1].theta):.2f}deg "
          f"kappa={traj[-1].kappa:.4f}")
    print()
    print(f"{'t':>5} {'x':>8} {'y':>8} {'v':>6} {'yaw':>7} "
          f"{'steer':>7} {'s_fb':>7} {'s_ff':>7} {'acc':>7} "
          f"{'lat_e':>8} {'hd(d)':>7}")
    every = max(1, len(hist) // 25)
    for h in hist[::every]:
        print(
            f"{h['t']:5.2f} {h['x']:8.2f} {h['y']:8.2f} "
            f"{h['v']:6.2f} {math.degrees(h['yaw']):7.2f} "
            f"{h['steer']:7.2f} {h['steer_fb']:7.2f} {h['steer_ff']:7.2f} "
            f"{h['acc']:7.3f} {h['lateral_error']:8.3f} "
            f"{math.degrees(h['heading_error']):7.2f}")
    print()


if __name__ == '__main__':
    cfg = load_config('configs/default.yaml')
    cfg['vehicle']['model_type'] = 'truck_trailer'

    # 圆弧 R=30m, v=5 m/s；arc_angle 约对应 20s 弧长 = v*T = 100m → 100/30 ≈ 3.33 rad
    import math as _m
    traj = generate_circle(radius=30.0, speed=5.0,
                           arc_angle=_m.pi * 2 / 3 * 2, dt=0.02)
    run_and_dump('CIRCLE R=30 v=5 (truck_trailer)', traj, cfg, 5.0)

    # 参照：同样场景但 kinematic plant，看是否同样发散
    cfg2 = load_config('configs/default.yaml')
    cfg2['vehicle']['model_type'] = 'kinematic'
    run_and_dump('CIRCLE R=30 v=5 (kinematic REF)', traj, cfg2, 5.0)
