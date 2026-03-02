# sim/run_demo.py
"""V1 可视化 Demo：4 种轨迹 × 4 张图 + 总览四宫格。

用法：
    python run_demo.py                   # 交互显示
    python run_demo.py --save            # 同时保存 PNG 到 sim/results/
    python run_demo.py --save --no-show  # 只保存，不弹窗
"""
import argparse
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

from trajectory import (generate_straight, generate_circle,
                        generate_sine, generate_combined)
from sim_loop import run_simulation

# ---------- 中文字体设置（优先微软雅黑，fallback SimHei）----------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 场景配色 ----------
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_scenario(name: str, history: list[dict], traj_pts) -> plt.Figure:
    """画 4 张子图：轨迹对比、横向误差、速度跟踪、转向角。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(name, fontsize=14)

    ts = [h['t'] for h in history]

    # 1. 轨迹对比
    ax = axes[0, 0]
    ax.plot([p.x for p in traj_pts], [p.y for p in traj_pts],
            'b--', label='参考轨迹', linewidth=1)
    ax.plot([h['x'] for h in history], [h['y'] for h in history],
            'r-', label='实际轨迹', linewidth=1)
    ax.set_xlabel('x 位置 (m)')
    ax.set_ylabel('y 位置 (m)')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('轨迹对比')
    ax.grid(True)

    # 2. 横向误差
    ax = axes[0, 1]
    ax.plot(ts, [h['lateral_error'] for h in history], 'g-',
            label='横向偏差')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('横向误差 (m)')
    ax.set_title('横向误差')
    ax.legend()
    ax.grid(True)

    # 3. 速度跟踪
    ax = axes[1, 0]
    ref_v = traj_pts[0].v
    ax.plot(ts, [h['v'] for h in history], 'r-', label='实际速度')
    ax.axhline(y=ref_v, color='b', linestyle='--',
               label=f'参考速度 {ref_v:.1f} m/s')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('速度跟踪')
    ax.legend()
    ax.grid(True)

    # 4. 转向角
    ax = axes[1, 1]
    ax.plot(ts, [h['steer'] for h in history], 'm-',
            label='方向盘转角')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('方向盘转角 (°)')
    ax.set_title('转向角输出')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_overview(all_results: list[dict]) -> plt.Figure:
    """四宫格总览：每个子图为一种工况的轨迹跟踪。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('V1 轨迹跟踪总览 — LatControllerTruck + LonController（简化版）',
                 fontsize=14)

    for idx, (ax, res) in enumerate(zip(axes.flat, all_results)):
        traj_pts = res['traj']
        history = res['history']
        name = res['name']
        color = _COLORS[idx]

        ax.plot([p.x for p in traj_pts], [p.y for p in traj_pts],
                'k--', label='参考轨迹', linewidth=1, alpha=0.7)
        ax.plot([h['x'] for h in history], [h['y'] for h in history],
                '-', color=color, label='实际轨迹', linewidth=1.5)
        ax.set_xlabel('x 位置 (m)')
        ax.set_ylabel('y 位置 (m)')
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='V1 Demo: 4 scenarios × 4 plots + overview')
    parser.add_argument('--save', action='store_true',
                        help='保存 PNG 到 sim/results/')
    parser.add_argument('--no-show', action='store_true',
                        help='不弹出交互窗口（配合 --save 使用）')
    args = parser.parse_args()

    # 非交互模式时切换后端
    if args.no_show:
        matplotlib.use('Agg')

    scenarios = [
        ('直线跟踪 (10 m/s)',
         generate_straight(length=200, speed=10.0), 10.0),
        ('圆弧跟踪 (R=30m, 5 m/s)',
         generate_circle(radius=30.0, speed=5.0, arc_angle=math.pi), 5.0),
        ('正弦曲线 (A=3m, λ=50m, 5 m/s)',
         generate_sine(amplitude=3.0, wavelength=50.0, n_waves=2, speed=5.0), 5.0),
        ('组合轨迹 (直线→弯→直线, 5 m/s)',
         generate_combined(speed=5.0), 5.0),
    ]

    # 保存目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if args.save:
        os.makedirs(results_dir, exist_ok=True)

    all_results = []
    safe_names = ['straight', 'circle', 'sine', 'combined']

    for i, (name, traj, init_v) in enumerate(scenarios):
        print(f"运行: {name} ...")
        history = run_simulation(traj, init_speed=init_v)
        all_results.append({'name': name, 'traj': traj, 'history': history})

        fig = plot_scenario(name, history, traj)
        if args.save:
            path = os.path.join(results_dir, f'{i+1}_{safe_names[i]}.png')
            fig.savefig(path, dpi=150)
            print(f"  已保存: {path}")

    # 总览四宫格
    fig_overview = plot_overview(all_results)
    if args.save:
        path = os.path.join(results_dir, '0_overview.png')
        fig_overview.savefig(path, dpi=150)
        print(f"  已保存: {path}")

    if not args.no_show:
        plt.show()
    else:
        print("完成。图片已保存（无交互窗口）。")


if __name__ == '__main__':
    main()
