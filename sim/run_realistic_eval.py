# sim/run_realistic_eval.py
"""园区实际工况前向仿真评估。

使用 hybrid_dynamic 模型 + default 参数（不训练），在 15 条园区工况轨迹上
运行前向仿真，保存轨迹跟踪和误差图到 sim/results/realistic_scenarios/。

用法：
    python run_realistic_eval.py                    # 运行仿真并保存图
    python run_realistic_eval.py --show             # 同时弹出交互窗口
    python run_realistic_eval.py --plant kinematic  # 改用运动学模型
"""
import argparse
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt

from config import load_config
from model.trajectory import (generate_straight, generate_lane_change,
                              generate_clothoid_turn, generate_uturn,
                              generate_stop_and_go, generate_park_route)
from sim_loop import run_simulation

# ---------- 中文字体 ----------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _to_float(val):
    if hasattr(val, 'item'):
        return val.item()
    return float(val)


def plot_scenario(name, history, traj_pts):
    """6 子图：轨迹对比、横向误差、速度跟踪、航向误差、转向角、加速度。"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(name, fontsize=14)

    ts = [_to_float(h['t']) for h in history]

    # 1. 轨迹对比
    ax = axes[0, 0]
    ax.plot([p.x for p in traj_pts], [p.y for p in traj_pts],
            'b--', label='参考轨迹', linewidth=1)
    ax.plot([_to_float(h['x']) for h in history],
            [_to_float(h['y']) for h in history],
            'r-', label='实际轨迹', linewidth=1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    ax.set_title('轨迹对比')
    ax.grid(True)

    # 2. 横向误差
    ax = axes[0, 1]
    lat_errs = [_to_float(h['lateral_error']) for h in history]
    ax.plot(ts, lat_errs, 'g-', label='横向偏差')
    rmse = (sum(e ** 2 for e in lat_errs) / len(lat_errs)) ** 0.5
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('横向误差 (m)')
    ax.set_title(f'横向误差 (RMSE={rmse:.4f} m)')
    ax.legend()
    ax.grid(True)

    # 3. 速度跟踪
    ax = axes[1, 0]
    actual_v = [_to_float(h['v']) for h in history]
    # 参考速度可能变化（stop_and_go），按时间插值
    ref_vs = []
    t_idx = 0
    for h in history:
        ht = _to_float(h['t'])
        while t_idx < len(traj_pts) - 2 and traj_pts[t_idx + 1].t < ht:
            t_idx += 1
        ref_vs.append(traj_pts[t_idx].v)
    ax.plot(ts, actual_v, 'r-', label='实际速度')
    ax.plot(ts, ref_vs, 'b--', label='参考速度')
    speed_errs = [a - r for a, r in zip(actual_v, ref_vs)]
    speed_rmse = (sum(e ** 2 for e in speed_errs) / len(speed_errs)) ** 0.5
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title(f'速度跟踪 (RMSE={speed_rmse:.4f} m/s)')
    ax.legend()
    ax.grid(True)

    # 4. 航向误差
    ax = axes[1, 1]
    head_errs = [math.degrees(_to_float(h['heading_error'])) for h in history]
    ax.plot(ts, head_errs, 'c-', label='航向偏差')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    head_rmse = (sum((e * math.pi / 180) ** 2 for e in head_errs)
                 / len(head_errs)) ** 0.5
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('航向误差 (°)')
    ax.set_title(f'航向误差 (RMSE={math.degrees(head_rmse):.4f}°)')
    ax.legend()
    ax.grid(True)

    # 5. 转向角
    ax = axes[2, 0]
    ax.plot(ts, [_to_float(h['steer']) for h in history], 'm-',
            label='总转角', linewidth=1.2)
    ax.plot(ts, [_to_float(h['steer_fb']) for h in history], 'b--',
            label='反馈', alpha=0.6, linewidth=0.8)
    ax.plot(ts, [_to_float(h['steer_ff']) for h in history], 'g:',
            label='前馈', alpha=0.6, linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('方向盘转角 (°)')
    ax.set_title('横向控制器输出 — 转向角')
    ax.legend(fontsize=9)
    ax.grid(True)

    # 6. 加速度
    ax = axes[2, 1]
    ax.plot(ts, [_to_float(h['acc']) for h in history], 'tab:orange',
            label='加速度指令')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('加速度 (m/s²)')
    ax.set_title('纵向控制器输出 — 加速度')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_overview(all_results):
    """总览：每个子图为一种工况的轨迹跟踪。"""
    n = len(all_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.suptitle('园区实际工况 — 轨迹跟踪总览', fontsize=14)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    flat_axes = axes.flat if hasattr(axes, 'flat') else [axes]
    for idx, res in enumerate(all_results):
        ax = flat_axes[idx]
        traj_pts = res['traj']
        history = res['history']
        name = res['name']
        color = colors[idx % len(colors)]

        ax.plot([p.x for p in traj_pts], [p.y for p in traj_pts],
                'k--', label='参考', linewidth=1, alpha=0.7)
        ax.plot([_to_float(h['x']) for h in history],
                [_to_float(h['y']) for h in history],
                '-', color=color, label='实际', linewidth=1.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        flat_axes[idx].set_visible(False)
    plt.tight_layout()
    return fig


def build_scenarios():
    """构建 15 条园区实际工况轨迹。"""
    scenarios = []

    # --- clothoid 右转 90° R=40m ---
    for v_kph in [10, 15]:
        v = v_kph / 3.6
        traj = generate_clothoid_turn(40.0, -math.pi / 2, v)
        scenarios.append((f'右转90° R=40m {v_kph}kph', traj, v,
                          f'clothoid_right90_R40_{v_kph}kph'))

    # --- clothoid 左转 90° R=35m ---
    for v_kph in [10, 15]:
        v = v_kph / 3.6
        traj = generate_clothoid_turn(35.0, math.pi / 2, v)
        scenarios.append((f'左转90° R=35m {v_kph}kph', traj, v,
                          f'clothoid_left90_R35_{v_kph}kph'))

    # --- clothoid 弯道 R=80m 45° ---
    for v_kph in [15, 25]:
        v = v_kph / 3.6
        traj = generate_clothoid_turn(80.0, math.pi / 4, v)
        scenarios.append((f'弯道45° R=80m {v_kph}kph', traj, v,
                          f'clothoid_curve45_R80_{v_kph}kph'))

    # --- 掉头 R=30m ---
    for v_kph in [5, 8]:
        v = v_kph / 3.6
        traj = generate_uturn(30.0, v)
        scenarios.append((f'掉头 R=30m {v_kph}kph', traj, v,
                          f'uturn_R30_{v_kph}kph'))

    # --- 换道 3.5m ---
    for v_kph in [15, 25]:
        v = v_kph / 3.6
        change_len = v * 5.0  # 约 5 秒换道时长
        traj = generate_lane_change(3.5, change_len, v)
        scenarios.append((f'换道3.5m {v_kph}kph', traj, v,
                          f'lane_change_{v_kph}kph'))

    # --- 停靠起步 ---
    for v_kph in [15, 25]:
        v = v_kph / 3.6
        traj = generate_stop_and_go(v, accel_rate=0.5, decel_rate=0.5)
        scenarios.append((f'停靠起步 {v_kph}kph', traj, v,
                          f'stop_and_go_{v_kph}kph'))

    # --- 综合园区路线 ---
    v = 15.0 / 3.6
    traj = generate_park_route(cruise_speed=v)
    scenarios.append(('综合园区路线 15kph', traj, v, 'park_route_15kph'))

    # --- 直道巡航 ---
    for v_kph in [20, 30]:
        v = v_kph / 3.6
        traj = generate_straight(200.0, v)
        scenarios.append((f'直道巡航 {v_kph}kph', traj, v,
                          f'straight_{v_kph}kph'))

    return scenarios


def main():
    parser = argparse.ArgumentParser(description='园区实际工况前向仿真评估')
    parser.add_argument('--show', action='store_true', help='弹出交互窗口')
    parser.add_argument('--plant', type=str, default='hybrid_dynamic',
                        choices=['kinematic', 'dynamic', 'hybrid_dynamic'],
                        help='被控对象类型（默认 hybrid_dynamic）')
    parser.add_argument('--config', type=str, default=None,
                        help='加载指定 YAML 配置')
    args = parser.parse_args()

    if not args.show:
        matplotlib.use('Agg')

    cfg = load_config(args.config) if args.config else load_config()
    cfg['vehicle']['model_type'] = args.plant

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results', 'realistic_scenarios', args.plant)
    os.makedirs(results_dir, exist_ok=True)

    scenarios = build_scenarios()
    all_results = []

    print(f"=== 园区实际工况评估 ({args.plant}) ===")
    print(f"共 {len(scenarios)} 条轨迹\n")

    for i, (name, traj, init_v, safe_name) in enumerate(scenarios):
        print(f"[{i+1:2d}/{len(scenarios)}] {name} ...", end=' ', flush=True)
        try:
            history = run_simulation(traj, init_speed=init_v, cfg=cfg)
            all_results.append({
                'name': name, 'traj': traj, 'history': history,
                'safe_name': safe_name})

            # 计算 RMSE
            lat_errs = [_to_float(h['lateral_error']) for h in history]
            lat_rmse = (sum(e ** 2 for e in lat_errs) / len(lat_errs)) ** 0.5

            fig = plot_scenario(name, history, traj)
            path = os.path.join(results_dir, f'{i+1:02d}_{safe_name}.png')
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f'lat_RMSE={lat_rmse:.4f}m  已保存')
        except Exception as e:
            print(f'失败: {e}')

    if all_results:
        fig_overview = plot_overview(all_results)
        path = os.path.join(results_dir, '00_overview.png')
        fig_overview.savefig(path, dpi=150)
        plt.close(fig_overview)
        print(f"\n总览图已保存: {path}")

    # 打印汇总
    print(f"\n=== 汇总 ({len(all_results)}/{len(scenarios)} 成功) ===")
    print(f"{'场景':<35s} {'lat_RMSE(m)':>12s} {'head_RMSE(°)':>13s} "
          f"{'speed_RMSE':>12s}")
    print('-' * 75)
    for res in all_results:
        history = res['history']
        traj = res['traj']
        lat_errs = [_to_float(h['lateral_error']) for h in history]
        head_errs = [_to_float(h['heading_error']) for h in history]
        actual_v = [_to_float(h['v']) for h in history]
        # 参考速度
        ref_vs = []
        t_idx = 0
        for h in history:
            ht = _to_float(h['t'])
            while t_idx < len(traj) - 2 and traj[t_idx + 1].t < ht:
                t_idx += 1
            ref_vs.append(traj[t_idx].v)

        lat_rmse = (sum(e ** 2 for e in lat_errs) / len(lat_errs)) ** 0.5
        head_rmse = math.degrees(
            (sum(e ** 2 for e in head_errs) / len(head_errs)) ** 0.5)
        speed_errs = [a - r for a, r in zip(actual_v, ref_vs)]
        speed_rmse = (sum(e ** 2 for e in speed_errs) / len(speed_errs)) ** 0.5

        print(f'{res["name"]:<35s} {lat_rmse:>12.4f} {head_rmse:>13.4f} '
              f'{speed_rmse:>12.4f}')

    print(f'\n结果保存在: {results_dir}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
