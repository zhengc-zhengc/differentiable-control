"""验证稳态误差成因的 4 个最小实验。
plant 固定为 truck_trailer，轨迹固定为 R=30 v=5 圆弧。
每组只改一个变量，对比最终 lat_err。
"""
import sys
import math
import copy
sys.path.insert(0, '.')
from config import load_config
from model.trajectory import generate_circle
from sim_loop import run_simulation


def run(cfg, name):
    traj = generate_circle(radius=30.0, speed=5.0,
                           arc_angle=math.pi * 1.5, dt=0.02)
    hist = run_simulation(traj, init_speed=5.0, cfg=cfg, differentiable=False)
    lat_errs = [abs(h['lateral_error']) for h in hist]
    hd_errs = [abs(h['heading_error']) for h in hist]
    v_last = hist[-1]['v']
    fb_avg = sum(h['steer_fb'] for h in hist[-50:]) / 50.0
    ff_avg = sum(h['steer_ff'] for h in hist[-50:]) / 50.0
    print(f"{name:55s}  "
          f"|lat_err|max={max(lat_errs):6.2f}m  last={lat_errs[-1]:6.2f}m  "
          f"|hd|max={math.degrees(max(hd_errs)):5.2f}°  "
          f"v_end={v_last:4.2f}  "
          f"fb_last50={fb_avg:6.1f}  ff={ff_avg:6.1f}")


def baseline_cfg():
    cfg = load_config('configs/default.yaml')
    cfg['vehicle']['model_type'] = 'truck_trailer'
    return cfg


if __name__ == '__main__':
    # A) 参照
    run(baseline_cfg(), 'A. truck_trailer baseline')

    # B) 去掉 T1 最大航向误差限幅（把 3.86° 抬到 45°）
    cfg = baseline_cfg()
    cfg['lat_truck']['T1_max_theta_deg'] = [[s, 45.0]
                                            for s in [0, 10, 20, 30, 40, 50, 60]]
    run(cfg, 'B. T1 clamp 3.86°→45° (uncap P term)')

    # C) 加大 P 增益：缩小 reach_time_theta（denom 变小，target_curvature 变大）
    cfg = baseline_cfg()
    cfg['lat_truck']['T3_reach_time_theta'] = [[s, 0.4]
                                               for s in [0, 10, 20, 30, 40, 50, 60]]
    run(cfg, 'C. T3 reach_time 1.1→0.4 (raise P gain)')

    # D) 同时做 B + C
    cfg = baseline_cfg()
    cfg['lat_truck']['T1_max_theta_deg'] = [[s, 45.0]
                                            for s in [0, 10, 20, 30, 40, 50, 60]]
    cfg['lat_truck']['T3_reach_time_theta'] = [[s, 0.4]
                                               for s in [0, 10, 20, 30, 40, 50, 60]]
    run(cfg, 'D. B+C (uncap P + raise P gain)')

    # E) kinematic 对照（plant ≡ FF 模型）
    cfg = baseline_cfg()
    cfg['vehicle']['model_type'] = 'kinematic'
    run(cfg, 'E. kinematic plant (reference)')

    # F) dynamic 对照（乘用车动力学，原调参目标）
    cfg = baseline_cfg()
    cfg['vehicle']['model_type'] = 'dynamic'
    run(cfg, 'F. dynamic plant (passenger car)')
