# sim/optim/validate_batch.py
"""批量并行 49 场景 V1 验证 / 任意 A/B 对比入口。

典型用法（与 post_training 等价的 baseline vs tuned V1 对比）：
    python optim/validate_batch.py --config configs/tuned/xxx.yaml --plant truck_trailer

自定义 A/B 对比（比如"有 MLP vs 无 MLP"）：
    python optim/validate_batch.py \\
        --config-a configs/default.yaml --label-a "有 MLP" \\
        --config-b /tmp/default_no_mlp.yaml --label-b "无 MLP" \\
        --plant truck_trailer \\
        --output-suffix mlp_ablation
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from model.trajectory import generate_park_route  # noqa: F401
from optim.post_training import _build_eval_scenarios, _calc_metrics
from optim.train_batch import run_simulation_batch

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _batched_to_scenario_histories(batched_hist: dict, n_scenarios: int,
                                    dt: float) -> list:
    """把 run_simulation_batch 的 [B, T_max] history 拆成 scalar-like list[dict] per scenario。

    兼容 post_training._plot_comparison_grid 的输入格式。
    每个 scenario 截断到 valid_mask=1 的步数。
    """
    mask = batched_hist['valid_mask']  # [B, T]
    scenarios = []
    for b in range(n_scenarios):
        n_valid = int(mask[b].sum().item())
        hist = []
        for t_idx in range(n_valid):
            hist.append({
                't': t_idx * dt,
                'x': float(batched_hist['x'][b, t_idx]),
                'y': float(batched_hist['y'][b, t_idx]),
                'yaw': float(batched_hist['yaw'][b, t_idx]),
                'v': float(batched_hist['v'][b, t_idx]),
                'steer': float(batched_hist['steer'][b, t_idx]),
                'steer_fb': float(batched_hist['steer_fb'][b, t_idx]),
                'steer_ff': float(batched_hist['steer_ff'][b, t_idx]),
                'acc': float(batched_hist['acc'][b, t_idx]),
                'lateral_error': float(batched_hist['lateral_error'][b, t_idx]),
                'heading_error': float(batched_hist['heading_error'][b, t_idx]),
                'ref_x': float(batched_hist['ref_x'][b, t_idx]),
                'ref_y': float(batched_hist['ref_y'][b, t_idx]),
            })
        scenarios.append(hist)
    return scenarios


def _plot_comparison_grid_custom(all_a, all_b, output_dir, plot_type, filename,
                                  label_a: str, label_b: str, title_prefix: str):
    """_plot_comparison_grid 的自定义 label 版，去除 hardcoded "调参前/后" 字样。"""
    titles_map = {
        'trajectory': f'{title_prefix}轨迹跟踪对比',
        'lateral_error': f'{title_prefix}横向误差对比',
        'speed_error': f'{title_prefix}纵向（速度）误差对比',
        'steer': f'{title_prefix}转向角输出对比',
        'acc': f'{title_prefix}加速度输出对比',
    }
    n = len(all_a)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(titles_map[plot_type], fontsize=16)
    flat_axes = axes.flat if hasattr(axes, 'flat') else [axes]

    for idx in range(n):
        ax = flat_axes[idx]
        key, name, traj, h_a, m_a, ref_v = all_a[idx]
        _, _, _, h_b, m_b, _ = all_b[idx]

        if plot_type == 'trajectory':
            ax.plot([p.x for p in traj], [p.y for p in traj],
                    'k--', label='参考轨迹', linewidth=1, alpha=0.7)
            ax.plot([h['x'] for h in h_a], [h['y'] for h in h_a],
                    'b-', label=f'{label_a} (lat={m_a["lat_rmse"]:.3f}m)',
                    linewidth=1.2, alpha=0.8)
            ax.plot([h['x'] for h in h_b], [h['y'] for h in h_b],
                    'r-', label=f'{label_b} (lat={m_b["lat_rmse"]:.3f}m)',
                    linewidth=1.2, alpha=0.8)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal', adjustable='datalim')
        elif plot_type == 'lateral_error':
            ax.plot([h['t'] for h in h_a], [h['lateral_error'] for h in h_a],
                    'b-', label=f'{label_a} (RMSE={m_a["lat_rmse"]:.3f}m)',
                    alpha=0.8)
            ax.plot([h['t'] for h in h_b], [h['lateral_error'] for h in h_b],
                    'r-', label=f'{label_b} (RMSE={m_b["lat_rmse"]:.3f}m)',
                    alpha=0.8)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('横向误差 (m)')
        elif plot_type == 'speed_error':
            err_a = [h['v'] - ref_v for h in h_a]
            err_b = [h['v'] - ref_v for h in h_b]
            rmse_a = (sum(e**2 for e in err_a) / len(err_a)) ** 0.5
            rmse_b = (sum(e**2 for e in err_b) / len(err_b)) ** 0.5
            ax.plot([h['t'] for h in h_a], err_a,
                    'b-', label=f'{label_a} (RMSE={rmse_a:.3f}m/s)', alpha=0.8)
            ax.plot([h['t'] for h in h_b], err_b,
                    'r-', label=f'{label_b} (RMSE={rmse_b:.3f}m/s)', alpha=0.8)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('速度误差 (m/s)')
        elif plot_type == 'steer':
            t_a = [h['t'] for h in h_a]
            t_b = [h['t'] for h in h_b]
            ax.plot(t_a, [h['steer'] for h in h_a],
                    'b-', label=f'{label_a} 总转角', alpha=0.8, linewidth=1.2)
            ax.plot(t_b, [h['steer'] for h in h_b],
                    'r-', label=f'{label_b} 总转角', alpha=0.8, linewidth=1.2)
            ax.plot(t_a, [h['steer_fb'] for h in h_a],
                    'b--', label=f'{label_a} 反馈', alpha=0.5, linewidth=0.8)
            ax.plot(t_a, [h['steer_ff'] for h in h_a],
                    'b:', label=f'{label_a} 前馈', alpha=0.5, linewidth=0.8)
            ax.plot(t_b, [h['steer_fb'] for h in h_b],
                    'r--', label=f'{label_b} 反馈', alpha=0.5, linewidth=0.8)
            ax.plot(t_b, [h['steer_ff'] for h in h_b],
                    'r:', label=f'{label_b} 前馈', alpha=0.5, linewidth=0.8)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('方向盘转角 (deg)')
        elif plot_type == 'acc':
            ax.plot([h['t'] for h in h_a], [h['acc'] for h in h_a],
                    'b-', label=label_a, alpha=0.8)
            ax.plot([h['t'] for h in h_b], [h['acc'] for h in h_b],
                    'r-', label=label_b, alpha=0.8)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('加速度 (m/s²)')

        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        flat_axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _apply_config_overrides(cfg: dict, trailer_mass_kg, disable_mlp: bool):
    """允许命令行覆盖挂车质量和 MLP 开关（基于已加载 cfg，修改 in place）。"""
    if trailer_mass_kg is not None:
        cfg['truck_trailer_vehicle']['default_trailer_mass_kg'] = float(trailer_mass_kg)
    if disable_mlp:
        cfg['truck_trailer_vehicle']['checkpoint_path'] = ''


def run_ab_validation(cfg_a: dict, cfg_b: dict, label_a: str, label_b: str,
                      output_dir: str, plant: str = None,
                      trajectory_types=None, verbose: bool = True,
                      title_prefix: str = '') -> dict:
    """并行跑 49 场景 V1 验证，对比两套 config 的跟踪性能。

    返回: scenario_key -> {label_a_metric, label_b_metric, delta_lat_pct,
                           delta_head_pct}
    """
    if plant:
        apply_plant_override(cfg_a, plant)
        apply_plant_override(cfg_b, plant)
    assert cfg_a['vehicle']['model_type'] == 'truck_trailer', \
        "validate_batch 目前仅支持 truck_trailer"
    assert cfg_b['vehicle']['model_type'] == 'truck_trailer'

    dt = cfg_a['simulation']['dt']

    # 构建 49 场景
    scenarios = _build_eval_scenarios(trajectory_types)
    trajs = [gen() for _key, _lbl, gen in scenarios]
    ref_speeds = [float(t[0].v) for t in trajs]
    n = len(trajs)

    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"{'='*60}")
        print(f"并行 V1 验证：{label_a} vs {label_b}")
        print(f"  场景数: {n}")
        print(f"  输出: {output_dir}")
        print(f"{'='*60}\n")

    # 两次 batched 仿真，每次 B=49 一次跑完
    t0 = time.time()
    hist_a_batch = run_simulation_batch(trajs, cfg=cfg_a, tbptt_k=0,
                                         hard_mode=True)
    dt_a = time.time() - t0
    if verbose:
        print(f"[{label_a}] 批量仿真完成，耗时 {dt_a:.1f}s")

    t0 = time.time()
    hist_b_batch = run_simulation_batch(trajs, cfg=cfg_b, tbptt_k=0,
                                         hard_mode=True)
    dt_b = time.time() - t0
    if verbose:
        print(f"[{label_b}] 批量仿真完成，耗时 {dt_b:.1f}s\n")

    # 拆成 per-scenario list[dict]
    hists_a = _batched_to_scenario_histories(hist_a_batch, n, dt)
    hists_b = _batched_to_scenario_histories(hist_b_batch, n, dt)

    # 计算 metrics
    all_a, all_b = [], []
    comparison_metrics = {}
    if verbose:
        print(f"{'场景':<35} {'lat RMSE(m)':>14} {'head RMSE(rad)':>16}")
        print('-' * 72)
    for i, ((key, name, _gen), traj, h_a, h_b) in enumerate(
            zip(scenarios, trajs, hists_a, hists_b)):
        m_a = _calc_metrics(h_a)
        m_b = _calc_metrics(h_b)
        ref_v = ref_speeds[i]
        all_a.append((key, name, traj, h_a, m_a, ref_v))
        all_b.append((key, name, traj, h_b, m_b, ref_v))

        d_lat = ((m_b['lat_rmse'] - m_a['lat_rmse']) / m_a['lat_rmse'] * 100
                 if m_a['lat_rmse'] > 0 else 0.0)
        d_head = ((m_b['head_rmse'] - m_a['head_rmse']) / m_a['head_rmse'] * 100
                  if m_a['head_rmse'] > 0 else 0.0)
        comparison_metrics[key] = {
            'label_a': label_a, 'label_b': label_b,
            'a_lat_rmse': m_a['lat_rmse'], 'b_lat_rmse': m_b['lat_rmse'],
            'a_head_rmse': m_a['head_rmse'], 'b_head_rmse': m_b['head_rmse'],
            'a_lat_max': m_a['lat_max'], 'b_lat_max': m_b['lat_max'],
            'delta_lat_pct': round(d_lat, 2),
            'delta_head_pct': round(d_head, 2),
        }
        if verbose:
            print(f"  [{i+1:2d}/{n}] {name:<30}"
                  f" {m_a['lat_rmse']:.4f}→{m_b['lat_rmse']:.4f} ({d_lat:+.1f}%)"
                  f"  {m_a['head_rmse']:.4f}→{m_b['head_rmse']:.4f} ({d_head:+.1f}%)")

    # 画图
    for plot_type, filename in [
            ('trajectory', 'comparison_trajectory.png'),
            ('lateral_error', 'comparison_lateral_error.png'),
            ('speed_error', 'comparison_speed_error.png'),
            ('steer', 'comparison_steer.png'),
            ('acc', 'comparison_acc.png')]:
        _plot_comparison_grid_custom(
            all_a, all_b, output_dir, plot_type, filename,
            label_a=label_a, label_b=label_b, title_prefix=title_prefix)

    # 汇总指标
    improved_lat = sum(1 for v in comparison_metrics.values()
                       if v['delta_lat_pct'] < 0)
    degraded_lat = sum(1 for v in comparison_metrics.values()
                       if v['delta_lat_pct'] > 0)
    avg_lat = sum(v['delta_lat_pct']
                  for v in comparison_metrics.values()) / len(comparison_metrics)
    avg_head = sum(v['delta_head_pct']
                   for v in comparison_metrics.values()) / len(comparison_metrics)

    summary = {
        'label_a': label_a, 'label_b': label_b,
        'n_scenarios': n,
        'improved_lat': improved_lat, 'degraded_lat': degraded_lat,
        'avg_lat_delta_pct': round(avg_lat, 2),
        'avg_head_delta_pct': round(avg_head, 2),
        'sim_time_a_s': round(dt_a, 1),
        'sim_time_b_s': round(dt_b, 1),
        'per_scenario': comparison_metrics,
    }

    with open(os.path.join(output_dir, 'validation_log.yaml'), 'w',
               encoding='utf-8') as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    if verbose:
        print()
        print(f"--- 验证汇总 ---")
        print(f"  场景数: {n} (lat 改善: {improved_lat}, 恶化: {degraded_lat})")
        print(f"  平均 lat RMSE 变化: {avg_lat:+.2f}%")
        print(f"  平均 head RMSE 变化: {avg_head:+.2f}%")
        print(f"  输出目录: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='并行 V1 验证（batched hard_mode）。支持 baseline vs tuned '
                    '或自定义 A/B 对比。')
    parser.add_argument('--config', type=str, default=None,
                        help='tuned 配置路径（传统 baseline vs tuned 模式）。'
                             '若提供则 A=default.yaml, B=--config。')
    parser.add_argument('--config-a', type=str, default=None,
                        help='A/B 模式下的 A 配置路径')
    parser.add_argument('--config-b', type=str, default=None,
                        help='A/B 模式下的 B 配置路径')
    parser.add_argument('--label-a', type=str, default='调参前',
                        help='A 的图例标签')
    parser.add_argument('--label-b', type=str, default='调参后',
                        help='B 的图例标签')
    parser.add_argument('--title-prefix', type=str, default='调参前后',
                        help='图标题前缀')
    parser.add_argument('--plant', type=str, default='truck_trailer',
                        choices=['truck_trailer'])
    parser.add_argument('--trailer-mass-a', type=float, default=None,
                        help='覆盖 A 的挂车质量 (kg)')
    parser.add_argument('--trailer-mass-b', type=float, default=None,
                        help='覆盖 B 的挂车质量 (kg)')
    parser.add_argument('--disable-mlp-a', action='store_true',
                        help='A 禁用 MLP 残差（清空 checkpoint_path）')
    parser.add_argument('--disable-mlp-b', action='store_true',
                        help='B 禁用 MLP 残差')
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='轨迹类型子集（默认全量 49 场景）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录，默认 results/validation/truck_trailer/<ts>_batch')
    parser.add_argument('--output-suffix', type=str, default='batch',
                        help='默认输出目录名后缀')
    args = parser.parse_args()

    # 解析 config 来源
    if args.config and (args.config_a or args.config_b):
        raise SystemExit("--config 与 --config-a/--config-b 互斥")
    if args.config:
        cfg_a = load_config(None)            # default.yaml
        cfg_b = load_config(args.config)
        label_a, label_b = args.label_a, args.label_b
        title_prefix = args.title_prefix
    else:
        if not (args.config_a and args.config_b):
            raise SystemExit("请提供 --config <tuned.yaml> 或同时提供 "
                             "--config-a / --config-b")
        cfg_a = load_config(args.config_a)
        cfg_b = load_config(args.config_b)
        label_a, label_b = args.label_a, args.label_b
        title_prefix = args.title_prefix

    _apply_config_overrides(cfg_a, args.trailer_mass_a, args.disable_mlp_a)
    _apply_config_overrides(cfg_b, args.trailer_mass_b, args.disable_mlp_b)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results', 'validation', args.plant,
            f'{ts}_{args.output_suffix}')

    run_ab_validation(cfg_a, cfg_b, label_a, label_b, output_dir,
                      plant=args.plant,
                      trajectory_types=args.trajectories,
                      title_prefix=title_prefix)


if __name__ == '__main__':
    main()
