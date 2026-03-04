# sim/compare_results.py
"""对比调参前后的跟踪性能指标 + 生成并排对比图。"""
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import load_config
from model.trajectory import (generate_straight, generate_circle,
                              generate_sine, generate_combined)
from sim_loop import run_simulation

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calc_metrics(history):
    """从 float 路径的 history 计算跟踪指标。"""
    lat = [abs(h['lateral_error']) for h in history]
    head = [abs(h['heading_error']) for h in history]
    lat_rmse = (sum(e**2 for e in lat) / len(lat)) ** 0.5
    head_rmse = (sum(e**2 for e in head) / len(head)) ** 0.5
    return {
        'lat_rmse': lat_rmse,
        'head_rmse': head_rmse,
        'lat_max': max(lat),
        'head_max': max(head),
    }


def main(tuned_config_path):
    scenarios = [
        ('straight', '直线 (10 m/s)',
         generate_straight(length=200, speed=10.0), 10.0),
        ('circle', '圆弧 (R=30m, 5 m/s)',
         generate_circle(radius=30.0, speed=5.0, arc_angle=math.pi), 5.0),
        ('sine', '正弦 (A=3m, 5 m/s)',
         generate_sine(amplitude=3.0, wavelength=50.0, n_waves=2, speed=5.0), 5.0),
        ('combined', '组合 (5 m/s)',
         generate_combined(speed=5.0), 5.0),
    ]

    cfg_base = load_config()
    cfg_tuned = load_config(tuned_config_path)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # --- 打印对比表格 ---
    header = f"{'Scenario':<25} {'':>8} {'lat_rmse(m)':>12} {'head_rmse(rad)':>14} {'lat_max(m)':>10}"
    print(header)
    print('-' * len(header))

    all_base = []
    all_tuned = []

    for key, name, traj, init_v in scenarios:
        h_base = run_simulation(traj, init_speed=init_v, cfg=cfg_base)
        h_tuned = run_simulation(traj, init_speed=init_v, cfg=cfg_tuned)
        m_base = calc_metrics(h_base)
        m_tuned = calc_metrics(h_tuned)

        all_base.append((key, name, traj, h_base, m_base))
        all_tuned.append((key, name, traj, h_tuned, m_tuned))

        print(f"{name:<25} {'baseline':>8} {m_base['lat_rmse']:>12.4f} "
              f"{m_base['head_rmse']:>14.4f} {m_base['lat_max']:>10.4f}")
        print(f"{'':25} {'tuned':>8} {m_tuned['lat_rmse']:>12.4f} "
              f"{m_tuned['head_rmse']:>14.4f} {m_tuned['lat_max']:>10.4f}")
        d_lat = ((m_tuned['lat_rmse'] - m_base['lat_rmse']) / m_base['lat_rmse'] * 100
                 if m_base['lat_rmse'] > 1e-8 else 0.0)
        d_head = ((m_tuned['head_rmse'] - m_base['head_rmse']) / m_base['head_rmse'] * 100
                  if m_base['head_rmse'] > 1e-8 else 0.0)
        print(f"{'':25} {'delta':>8} {d_lat:>+11.1f}% {d_head:>+13.1f}%")
        print()

    # --- 生成并排对比图（每个场景一张：左=baseline, 右=tuned） ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('调参前后轨迹跟踪对比', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        key_b, name_b, traj_b, h_b, m_b = all_base[idx]
        key_t, name_t, traj_t, h_t, m_t = all_tuned[idx]

        # 参考轨迹
        ax.plot([p.x for p in traj_b], [p.y for p in traj_b],
                'k--', label='参考轨迹', linewidth=1, alpha=0.7)
        # Baseline
        ax.plot([h['x'] for h in h_b], [h['y'] for h in h_b],
                'b-', label=f'调参前 (lat={m_b["lat_rmse"]:.3f}m)', linewidth=1.2, alpha=0.8)
        # Tuned
        ax.plot([h['x'] for h in h_t], [h['y'] for h in h_t],
                'r-', label=f'调参后 (lat={m_t["lat_rmse"]:.3f}m)', linewidth=1.2, alpha=0.8)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=9)
        ax.set_title(name_b)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_trajectory.png')
    fig.savefig(path, dpi=150)
    print(f"轨迹对比图已保存: {path}")

    # --- 横向误差对比图 ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('调参前后横向误差对比', fontsize=16)

    for idx, ax in enumerate(axes2.flat):
        key_b, name_b, _, h_b, m_b = all_base[idx]
        key_t, name_t, _, h_t, m_t = all_tuned[idx]

        ts_b = [h['t'] for h in h_b]
        ts_t = [h['t'] for h in h_t]

        ax.plot(ts_b, [h['lateral_error'] for h in h_b],
                'b-', label=f'调参前 (RMSE={m_b["lat_rmse"]:.3f}m)', alpha=0.8)
        ax.plot(ts_t, [h['lateral_error'] for h in h_t],
                'r-', label=f'调参后 (RMSE={m_t["lat_rmse"]:.3f}m)', alpha=0.8)

        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('横向误差 (m)')
        ax.set_title(name_b)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(results_dir, 'comparison_lateral_error.png')
    fig2.savefig(path2, dpi=150)
    print(f"横向误差对比图已保存: {path2}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <tuned_config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
