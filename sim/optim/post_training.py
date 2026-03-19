# sim/optim/post_training.py
"""训练后自动化：生成 loss 曲线、对比图、实验日志。"""
import math
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config, _get_commit_hash, _tensor_to_python, apply_plant_override
from model.trajectory import (expand_trajectories, generate_park_route,
                              TRAJECTORY_TYPES, SPEED_BANDS_KPH)
from sim_loop import run_simulation

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_loss_curves(training_history, output_dir):
    """绘制总 loss + 各加权分项随 epoch 变化的曲线。

    Args:
        training_history: train() 返回的 training_history 列表
        output_dir: 输出目录路径

    Returns:
        保存的文件路径
    """
    epochs = [r['epoch'] for r in training_history]
    total_loss = [r['loss'] for r in training_history]

    components = ['loss_lat', 'loss_head', 'loss_speed', 'loss_steer_rate', 'loss_acc_rate']
    labels = ['横向误差', '航向误差', '速度误差', '转向变化率', '加速度变化率']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 上图：总 loss
    ax1.plot(epochs, total_loss, 'k-', linewidth=2, label='总 loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'训练 Loss 曲线 ({total_loss[0]:.4f} -> {total_loss[-1]:.4f}, '
                  f'{(total_loss[-1]-total_loss[0])/total_loss[0]*100:+.1f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：各分项
    for comp, label, color in zip(components, labels, colors):
        vals = [r['avg'].get(comp, 0.0) for r in training_history]
        ax2.plot(epochs, vals, '-', color=color, label=label, linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss 分项')
    ax2.set_title('Loss 分项构成')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loss_breakdown(training_history, trajectories, output_dir):
    """绘制分轨迹的 loss 分项曲线（每条轨迹一个子图）。

    Args:
        training_history: train() 返回的 training_history 列表
        trajectories: 轨迹名列表
        output_dir: 输出目录路径

    Returns:
        保存的文件路径
    """
    n_traj = len(trajectories)
    fig, axes = plt.subplots(1, n_traj, figsize=(6 * n_traj, 5))
    if n_traj == 1:
        axes = [axes]
    fig.suptitle('分轨迹 Loss 分项', fontsize=14)

    components = ['loss_lat', 'loss_head', 'loss_speed']
    labels = ['横向', '航向', '速度']
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    epochs = [r['epoch'] for r in training_history]

    for idx, (ax, tname) in enumerate(zip(axes, trajectories)):
        for comp, label, color in zip(components, labels, colors):
            vals = [r['per_trajectory'][tname].get(comp, 0.0)
                    for r in training_history]
            ax.plot(epochs, vals, '-', color=color, label=label, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(tname)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'loss_breakdown.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _calc_metrics(history):
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


def _build_eval_scenarios(trajectory_types=None):
    """构建验证场景列表：类型×速度段 + park_route。

    Args:
        trajectory_types: 轨迹类型名列表，None 则使用全部 TRAJECTORY_TYPES

    Returns:
        [(key, label, generator, init_speed), ...]
    """
    expanded = expand_trajectories(trajectory_types)
    scenarios = []
    for key, label, gen in expanded:
        # 从 generator 获取 init_speed（生成后取首点速度）
        scenarios.append((key, label, gen))

    # 始终追加 park_route
    scenarios.append(('park_route', '园区综合', generate_park_route))
    return scenarios


def get_scenario_keys(trajectory_types=None):
    """返回所有可用场景的 key 列表。"""
    return [key for key, _, _ in _build_eval_scenarios(trajectory_types)]


def run_comparison(tuned_config_path, output_dir, verbose=True, plant=None,
                   trajectory_types=None):
    """用 V1 路径（float）跑 baseline vs tuned 对比，生成对比图 + 返回指标。

    Args:
        tuned_config_path: 调参后的配置文件路径
        output_dir: 输出目录路径
        verbose: 是否打印指标表格
        plant: 被控对象类型 ('kinematic'/'dynamic')，None 使用配置默认值
        trajectory_types: 轨迹类型名列表，None 则使用全部（8×6 + park_route = 49）

    Returns:
        comparison_metrics: {scenario_key: {baseline, tuned, delta_lat_pct, delta_head_pct}}
    """
    cfg_base = load_config()
    cfg_tuned = load_config(tuned_config_path)
    if plant:
        apply_plant_override(cfg_base, plant)
        apply_plant_override(cfg_tuned, plant)

    eval_scenarios = _build_eval_scenarios(trajectory_types)

    all_base = []
    all_tuned = []
    # key → label 映射，用于图表
    scenario_labels = {}

    if verbose:
        n_total = len(eval_scenarios)
        header = f"{'场景':<35} {'':>8} {'lat_rmse(m)':>12} {'head_rmse(rad)':>14} {'lat_max(m)':>10}"
        print(header)
        print('-' * len(header))

    comparison_metrics = {}

    for idx, (key, name, traj_gen) in enumerate(eval_scenarios):
        traj = traj_gen()
        init_v = traj[0].v
        scenario_labels[key] = name
        if verbose:
            print(f"  [{idx+1}/{len(eval_scenarios)}] {name}...", end='', flush=True)
        h_base = run_simulation(traj, init_speed=init_v, cfg=cfg_base)
        h_tuned = run_simulation(traj, init_speed=init_v, cfg=cfg_tuned)
        m_base = _calc_metrics(h_base)
        m_tuned = _calc_metrics(h_tuned)

        all_base.append((key, name, traj, h_base, m_base, init_v))
        all_tuned.append((key, name, traj, h_tuned, m_tuned, init_v))

        d_lat = ((m_tuned['lat_rmse'] - m_base['lat_rmse']) / m_base['lat_rmse'] * 100
                 if m_base['lat_rmse'] > 1e-8 else 0.0)
        d_head = ((m_tuned['head_rmse'] - m_base['head_rmse']) / m_base['head_rmse'] * 100
                  if m_base['head_rmse'] > 1e-8 else 0.0)

        comparison_metrics[key] = {
            'baseline': m_base,
            'tuned': m_tuned,
            'delta_lat_pct': round(d_lat, 2),
            'delta_head_pct': round(d_head, 2),
        }

        if verbose:
            sign_lat = '+' if d_lat > 0 else ''
            sign_head = '+' if d_head > 0 else ''
            print(f" lat: {m_base['lat_rmse']:.4f}→{m_tuned['lat_rmse']:.4f} "
                  f"({sign_lat}{d_lat:.1f}%) "
                  f"head: {m_base['head_rmse']:.4f}→{m_tuned['head_rmse']:.4f} "
                  f"({sign_head}{d_head:.1f}%)")

    # 5 种对比图
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='trajectory', filename='comparison_trajectory.png')
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='lateral_error', filename='comparison_lateral_error.png')
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='speed_error', filename='comparison_speed_error.png')
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='steer', filename='comparison_steer.png')
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='acc', filename='comparison_acc.png')

    return comparison_metrics, scenario_labels


def _plot_comparison_grid(all_base, all_tuned, output_dir, plot_type, filename):
    """通用对比图生成器，自适应场景数量。"""
    titles_map = {
        'trajectory': '调参前后轨迹跟踪对比',
        'lateral_error': '调参前后横向误差对比',
        'speed_error': '调参前后纵向（速度）误差对比',
        'steer': '调参前后转向角输出对比',
        'acc': '调参前后加速度输出对比',
    }
    n = len(all_base)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(titles_map[plot_type], fontsize=16)
    flat_axes = axes.flat if hasattr(axes, 'flat') else [axes]

    for idx in range(n):
        ax = flat_axes[idx]
        key_b, name_b, traj_b, h_b, m_b, ref_v = all_base[idx]
        _, _, _, h_t, m_t, _ = all_tuned[idx]

        if plot_type == 'trajectory':
            ax.plot([p.x for p in traj_b], [p.y for p in traj_b],
                    'k--', label='参考轨迹', linewidth=1, alpha=0.7)
            ax.plot([h['x'] for h in h_b], [h['y'] for h in h_b],
                    'b-', label=f'调参前 (lat={m_b["lat_rmse"]:.3f}m)',
                    linewidth=1.2, alpha=0.8)
            ax.plot([h['x'] for h in h_t], [h['y'] for h in h_t],
                    'r-', label=f'调参后 (lat={m_t["lat_rmse"]:.3f}m)',
                    linewidth=1.2, alpha=0.8)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal', adjustable='datalim')
        elif plot_type == 'lateral_error':
            ax.plot([h['t'] for h in h_b], [h['lateral_error'] for h in h_b],
                    'b-', label=f'调参前 (RMSE={m_b["lat_rmse"]:.3f}m)', alpha=0.8)
            ax.plot([h['t'] for h in h_t], [h['lateral_error'] for h in h_t],
                    'r-', label=f'调参后 (RMSE={m_t["lat_rmse"]:.3f}m)', alpha=0.8)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('横向误差 (m)')
        elif plot_type == 'speed_error':
            spd_err_b = [h['v'] - ref_v for h in h_b]
            spd_err_t = [h['v'] - ref_v for h in h_t]
            spd_rmse_b = (sum(e**2 for e in spd_err_b) / len(spd_err_b)) ** 0.5
            spd_rmse_t = (sum(e**2 for e in spd_err_t) / len(spd_err_t)) ** 0.5
            ax.plot([h['t'] for h in h_b], spd_err_b,
                    'b-', label=f'调参前 (RMSE={spd_rmse_b:.3f}m/s)', alpha=0.8)
            ax.plot([h['t'] for h in h_t], spd_err_t,
                    'r-', label=f'调参后 (RMSE={spd_rmse_t:.3f}m/s)', alpha=0.8)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('速度误差 (m/s)')
        elif plot_type == 'steer':
            t_b = [h['t'] for h in h_b]
            t_t = [h['t'] for h in h_t]
            ax.plot(t_b, [h['steer'] for h in h_b],
                    'b-', label='调参前 总转角', alpha=0.8, linewidth=1.2)
            ax.plot(t_t, [h['steer'] for h in h_t],
                    'r-', label='调参后 总转角', alpha=0.8, linewidth=1.2)
            ax.plot(t_b, [h['steer_fb'] for h in h_b],
                    'b--', label='调参前 反馈', alpha=0.5, linewidth=0.8)
            ax.plot(t_b, [h['steer_ff'] for h in h_b],
                    'b:', label='调参前 前馈', alpha=0.5, linewidth=0.8)
            ax.plot(t_t, [h['steer_fb'] for h in h_t],
                    'r--', label='调参后 反馈', alpha=0.5, linewidth=0.8)
            ax.plot(t_t, [h['steer_ff'] for h in h_t],
                    'r:', label='调参后 前馈', alpha=0.5, linewidth=0.8)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('方向盘转角 (deg)')
        elif plot_type == 'acc':
            ax.plot([h['t'] for h in h_b], [h['acc'] for h in h_b],
                    'b-', label='调参前', alpha=0.8)
            ax.plot([h['t'] for h in h_t], [h['acc'] for h in h_t],
                    'r-', label='调参后', alpha=0.8)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('加速度 (m/s^2)')

        ax.set_title(name_b)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(n, nrows * ncols):
        flat_axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_experiment_log(train_result, comparison_metrics, output_dir,
                        hyperparams, tuned_config_path):
    """保存实验日志 YAML。

    Args:
        train_result: train() 返回值
        comparison_metrics: run_comparison() 返回的对比指标
        output_dir: 输出目录
        hyperparams: 训练超参数 dict
        tuned_config_path: 调参配置文件路径

    Returns:
        保存的文件路径
    """
    commit = _get_commit_hash()
    log = {
        'commit': commit,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tuned_config': tuned_config_path,
        'hyperparams': hyperparams,
        'results': {
            'initial_loss': train_result['losses'][0],
            'final_loss': train_result['losses'][-1],
            'loss_change_pct': round(
                (train_result['losses'][-1] - train_result['losses'][0])
                / train_result['losses'][0] * 100, 2),
        },
        'comparison': {},
        'parameter_changes': {},
    }

    # 对比指标（转为 plain dict 以便 YAML 序列化）
    for key, val in comparison_metrics.items():
        log['comparison'][key] = {
            'baseline_lat_rmse': val['baseline']['lat_rmse'],
            'tuned_lat_rmse': val['tuned']['lat_rmse'],
            'delta_lat_pct': val['delta_lat_pct'],
            'baseline_head_rmse': val['baseline']['head_rmse'],
            'tuned_head_rmse': val['tuned']['head_rmse'],
            'delta_head_pct': val['delta_head_pct'],
        }

    # 参数变化
    for name in train_result['initial_params']:
        init_v = train_result['initial_params'][name]
        final_v = train_result['final_params'][name]
        if isinstance(init_v, (int, float)):
            delta = final_v - init_v
            pct = delta / max(abs(init_v), 1e-8) * 100
            log['parameter_changes'][name] = {
                'initial': round(float(init_v), 6),
                'final': round(float(final_v), 6),
                'delta': round(float(delta), 6),
                'delta_pct': round(float(pct), 2),
            }
        else:
            # 查找表：记录最大变化
            max_delta = max(abs(f - i) for f, i in zip(final_v, init_v))
            log['parameter_changes'][name] = {
                'max_delta': round(float(max_delta), 6),
                'final_range': [round(float(min(final_v)), 4),
                                round(float(max(final_v)), 4)],
            }

    path = os.path.join(output_dir, 'experiment_log.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(_tensor_to_python(log), f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    return path


def plot_training_summary(train_result, comparison_metrics, hyperparams, output_dir,
                          scenario_labels=None):
    """训练摘要仪表板：上半表格（超参+指标对比），下半柱状图（lat_rmse 变化%）。"""
    n_scenarios = len(comparison_metrics)
    fig_h = max(14, 12 + (n_scenarios - 8) * 0.3)  # 场景多时增大高度
    fig = plt.figure(figsize=(16, fig_h))
    fig.suptitle('训练摘要', fontsize=16, fontweight='bold')

    # ── 上半部分：两个表格并排 ──
    # 左表：训练超参 + loss 变化
    ax_info = fig.add_axes([0.05, 0.55, 0.42, 0.38])
    ax_info.axis('off')
    ax_info.set_title('训练配置与结果', fontsize=12, pad=10)

    loss_0 = train_result['losses'][0]
    loss_n = train_result['losses'][-1]
    loss_pct = (loss_n - loss_0) / loss_0 * 100

    traj_types = hyperparams.get('trajectory_types', [])
    traj_str = ', '.join(traj_types) if isinstance(traj_types, list) else str(traj_types)
    if len(traj_str) > 40:
        traj_str = traj_str[:37] + '...'

    info_data = [
        ['Epochs', str(hyperparams.get('epochs', '?'))],
        ['学习率', f"{hyperparams.get('lr', '?')}"],
        ['轨迹类型', traj_str],
        ['速度段 (kph)', ', '.join(str(s) for s in SPEED_BANDS_KPH)],
        ['TBPTT-K', str(hyperparams.get('tbptt_k', '?'))],
        ['梯度裁剪', str(hyperparams.get('grad_clip', '?'))],
        ['初始 Loss', f"{loss_0:.4f}"],
        ['最终 Loss', f"{loss_n:.4f}"],
        ['Loss 变化', f"{loss_pct:+.2f}%"],
    ]
    table1 = ax_info.table(cellText=info_data, colLabels=['项目', '值'],
                           loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.0, 1.4)
    # 表头着色
    for j in range(2):
        table1[0, j].set_facecolor('#4472C4')
        table1[0, j].set_text_props(color='white', fontweight='bold')
    # loss 变化行着色
    color = '#C6EFCE' if loss_pct <= 0 else '#FFC7CE'
    table1[len(info_data), 0].set_facecolor(color)
    table1[len(info_data), 1].set_facecolor(color)

    # 右表：各场景 baseline vs tuned 对比
    ax_comp = fig.add_axes([0.52, 0.55, 0.45, 0.38])
    ax_comp.axis('off')
    ax_comp.set_title('V1 路径验证（baseline vs tuned）', fontsize=12, pad=10)

    # 从 scenario_labels 获取显示名（由 run_comparison 传入或重建）
    scenario_names = scenario_labels or {k: k for k in comparison_metrics}
    comp_data = []
    for key in comparison_metrics:
        cm = comparison_metrics[key]
        comp_data.append([
            scenario_names.get(key, key),
            f"{cm['baseline']['lat_rmse']:.4f}",
            f"{cm['tuned']['lat_rmse']:.4f}",
            f"{cm['delta_lat_pct']:+.2f}%",
            f"{cm['baseline']['head_rmse']:.4f}",
            f"{cm['tuned']['head_rmse']:.4f}",
            f"{cm['delta_head_pct']:+.2f}%",
        ])

    if comp_data:
        col_labels = ['场景', 'base_lat', 'tune_lat', 'Δlat%',
                       'base_head', 'tune_head', 'Δhead%']
        table2 = ax_comp.table(cellText=comp_data, colLabels=col_labels,
                               loc='center', cellLoc='center')
        table2.auto_set_font_size(False)
        font_sz = 7 if n_scenarios > 12 else 9
        row_h = 1.2 if n_scenarios > 12 else 1.4
        table2.set_fontsize(font_sz)
        table2.scale(1.0, row_h)
        for j in range(len(col_labels)):
            table2[0, j].set_facecolor('#4472C4')
            table2[0, j].set_text_props(color='white', fontweight='bold')
        # delta 列着色
        for i, row in enumerate(comp_data):
            for col_idx in [3, 6]:  # Δlat%, Δhead%
                val = float(row[col_idx].rstrip('%'))
                color = '#C6EFCE' if val <= 0 else '#FFC7CE'
                table2[i + 1, col_idx].set_facecolor(color)

    # ── 下半部分：lat_rmse 变化百分比柱状图 ──
    ax_bar = fig.add_axes([0.1, 0.08, 0.8, 0.38])

    bar_keys = list(comparison_metrics.keys())
    bar_labels = [scenario_names.get(k, k) for k in bar_keys]
    lat_deltas = [comparison_metrics[k]['delta_lat_pct'] for k in bar_keys]
    head_deltas = [comparison_metrics[k]['delta_head_pct'] for k in bar_keys]

    x = np.arange(len(bar_keys))
    width = 0.35
    colors_lat = ['#2ca02c' if d <= 0 else '#d62728' for d in lat_deltas]
    colors_head = ['#1f77b4' if d <= 0 else '#ff7f0e' for d in head_deltas]

    bars1 = ax_bar.bar(x - width/2, lat_deltas, width, label='lat_rmse 变化%',
                       color=colors_lat, edgecolor='black', linewidth=0.5)
    bars2 = ax_bar.bar(x + width/2, head_deltas, width, label='head_rmse 变化%',
                       color=colors_head, edgecolor='black', linewidth=0.5)

    # 数值标签
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h,
                    f'{h:+.1f}%', ha='center',
                    va='bottom' if h >= 0 else 'top',
                    fontsize=7 if n_scenarios > 12 else 9)

    ax_bar.set_xticks(x)
    tick_fs = 8 if n_scenarios > 12 else 11
    ax_bar.set_xticklabels(bar_labels, fontsize=tick_fs, rotation=45, ha='right')
    ax_bar.set_ylabel('变化百分比 (%)')
    ax_bar.set_title('各场景跟踪精度变化（负值=改善）', fontsize=12)
    ax_bar.axhline(y=0, color='k', linewidth=0.8)
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, alpha=0.3, axis='y')

    path = os.path.join(output_dir, 'training_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_parameter_changes(train_result, output_dir):
    """参数变化可视化：标量参数表格 + 查找表训练前后热力图对比。"""
    initial = train_result['initial_params']
    final = train_result['final_params']

    # 分离标量参数和查找表参数
    scalar_names, scalar_init, scalar_final = [], [], []
    table_names = []
    table_init_vals, table_final_vals = [], []
    table_x_labels = None

    for name in initial:
        init_v = initial[name]
        final_v = final[name]
        short = name.replace('lon_ctrl.', '').replace('lat_ctrl.', '')
        if isinstance(init_v, (int, float)):
            scalar_names.append(short)
            scalar_init.append(init_v)
            scalar_final.append(final_v)
        else:
            table_names.append(short)
            table_init_vals.append(list(init_v))
            table_final_vals.append(list(final_v))
            if table_x_labels is None:
                table_x_labels = [f'{i * 10}' for i in range(len(init_v))]

    has_scalar = bool(scalar_names)
    has_table = bool(table_names)
    if not has_scalar and not has_table:
        return None

    # 布局：标量表格 + 查找表双热力图（训练前/训练后）
    n_rows = (1 if has_scalar else 0) + (1 if has_table else 0)
    height = (4 if has_scalar else 0) + (5 if has_table else 0) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, height))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle('参数变化可视化', fontsize=16, fontweight='bold')

    ax_idx = 0

    # ── 标量参数表格 ──
    if has_scalar:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.axis('off')
        ax.set_title('纵向标量参数（训练前 → 训练后）', fontsize=12, pad=10)

        cell_data = []
        for name, iv, fv in zip(scalar_names, scalar_init, scalar_final):
            delta = fv - iv
            pct = delta / max(abs(iv), 1e-8) * 100
            cell_data.append([
                name,
                f'{iv:.6f}',
                f'{fv:.6f}',
                f'{delta:+.6f}',
                f'{pct:+.2f}%',
            ])

        col_labels = ['参数', '训练前', '训练后', '变化量', '变化%']
        table_obj = ax.table(cellText=cell_data, colLabels=col_labels,
                             loc='center', cellLoc='center')
        table_obj.auto_set_font_size(False)
        table_obj.set_fontsize(10)
        table_obj.scale(1.0, 1.4)
        # 表头着色
        for j in range(len(col_labels)):
            table_obj[0, j].set_facecolor('#4472C4')
            table_obj[0, j].set_text_props(color='white', fontweight='bold')
        # 变化量着色：变化大的高亮
        for i, row in enumerate(cell_data):
            pct_val = float(row[4].rstrip('%'))
            if abs(pct_val) > 1.0:
                color = '#C6EFCE' if pct_val < 0 else '#FFC7CE'
                table_obj[i + 1, 3].set_facecolor(color)
                table_obj[i + 1, 4].set_facecolor(color)

    # ── 查找表双热力图 ──
    if has_table:
        ax = axes[ax_idx]
        ax.axis('off')
        ax.set_title('横向查找表参数（训练前 → 训练后）', fontsize=12, pad=10)

        data_init = np.array(table_init_vals)
        data_final = np.array(table_final_vals)
        n_tables, n_bp = data_init.shape

        # 共享色标范围
        vmin = min(data_init.min(), data_final.min())
        vmax = max(data_init.max(), data_final.max())

        # 用 GridSpec 在该 axes 区域内创建两个子热力图
        gs = fig.add_gridspec(1, 2, left=0.08, right=0.88, bottom=0.05 if not has_scalar else 0.05,
                              top=0.42 if has_scalar else 0.85, wspace=0.15)
        ax.set_visible(False)  # 隐藏占位 axes

        for sub_idx, (data, subtitle) in enumerate([(data_init, '训练前'), (data_final, '训练后')]):
            sub_ax = fig.add_subplot(gs[0, sub_idx])
            im = sub_ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
            sub_ax.set_xticks(np.arange(n_bp))
            sub_ax.set_xticklabels(table_x_labels or [str(i) for i in range(n_bp)], fontsize=9)
            sub_ax.set_xlabel('速度断点 (km/h)')
            sub_ax.set_yticks(np.arange(n_tables))
            sub_ax.set_yticklabels(table_names, fontsize=10)
            sub_ax.set_title(subtitle, fontsize=11)

            # 标注绝对值
            for i in range(n_tables):
                for j in range(n_bp):
                    val = data[i, j]
                    text_color = 'white' if val > (vmin + vmax) / 2 else 'black'
                    sub_ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                                fontsize=8, color=text_color)

        # 共享色条
        cbar_ax = fig.add_axes([0.90, 0.05 if not has_scalar else 0.05,
                                0.02, 0.37 if has_scalar else 0.80])
        fig.colorbar(im, cax=cbar_ax, label='参数值')

    path = os.path.join(output_dir, 'parameter_changes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def run_validation(tuned_config_path, output_dir=None, verbose=True,
                    plant=None, trajectory_types=None):
    """独立验证入口：仅跑 V1 对比 + 生成对比图，不需要 train_result。

    Args:
        tuned_config_path: 调参后的配置文件路径
        output_dir: 输出目录（None 则自动生成 results/validation/{plant}/{timestamp}/）
        verbose: 是否打印进度
        plant: 被控对象类型，None 使用配置默认值
        trajectory_types: 轨迹类型名列表，None 则全量验证

    Returns:
        output_dir: 产物保存目录路径
    """
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        plant_name = plant or 'kinematic'
        output_dir = _ensure_dir(os.path.join(sim_dir, 'results', 'validation',
                                              plant_name, timestamp))

    _ensure_dir(output_dir)
    eval_scenarios = _build_eval_scenarios(trajectory_types)

    if verbose:
        types_str = ', '.join(trajectory_types) if trajectory_types else '全部'
        print(f"\n{'='*60}")
        print(f"独立验证 — 配置: {tuned_config_path}")
        print(f"  类型: {types_str}")
        print(f"  场景数: {len(eval_scenarios)} (含 park_route)")
        print(f"  产物保存到: {output_dir}")
        print(f"{'='*60}\n")

    comparison_metrics, scenario_labels = run_comparison(
        tuned_config_path, output_dir, verbose=verbose, plant=plant,
        trajectory_types=trajectory_types)

    # 复制 tuned config 到产物目录
    shutil.copy2(tuned_config_path,
                 os.path.join(output_dir, os.path.basename(tuned_config_path)))

    if verbose:
        # 汇总统计
        n_improved = sum(1 for v in comparison_metrics.values()
                         if v['delta_lat_pct'] < 0)
        n_degraded = sum(1 for v in comparison_metrics.values()
                         if v['delta_lat_pct'] > 0)
        avg_lat = (sum(v['delta_lat_pct'] for v in comparison_metrics.values())
                   / len(comparison_metrics))
        avg_head = (sum(v['delta_head_pct'] for v in comparison_metrics.values())
                    / len(comparison_metrics))
        print(f"\n--- 验证汇总 ---")
        print(f"  场景数: {len(comparison_metrics)} "
              f"(改善: {n_improved}, 退化: {n_degraded})")
        print(f"  平均 lat_rmse 变化: {avg_lat:+.2f}%")
        print(f"  平均 head_rmse 变化: {avg_head:+.2f}%")
        print(f"  产物目录: {output_dir}")

    return output_dir


def run_post_training(train_result, hyperparams, verbose=True, plant=None,
                      trajectory_types=None):
    """训练后一站式自动化入口。

    Args:
        train_result: train() 的返回值
        hyperparams: 训练超参数 dict
        verbose: 是否打印进度
        plant: 被控对象类型 ('kinematic'/'dynamic')，None 使用配置默认值
        trajectory_types: 验证用轨迹类型，None 则全量

    Returns:
        output_dir: 产物保存目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    plant_name = plant or 'kinematic'
    output_dir = _ensure_dir(os.path.join(sim_dir, 'results', 'training',
                                          plant_name, timestamp))

    if verbose:
        print(f"\n{'='*60}")
        print(f"训练后自动化 — 产物保存到: {output_dir}")
        print(f"{'='*60}")

    # 1. Loss 曲线
    p = plot_loss_curves(train_result['training_history'], output_dir)
    if verbose:
        print(f"  Loss 曲线: {p}")

    # 2. 分轨迹 loss 分项
    traj_keys = train_result.get('trajectory_keys', [])
    p = plot_loss_breakdown(train_result['training_history'], traj_keys, output_dir)
    if verbose:
        print(f"  Loss 分项: {p}")

    # 3. V1 路径对比
    if verbose:
        print(f"\n  --- V1 路径验证（baseline vs tuned）---")
    comparison_metrics, scenario_labels = run_comparison(
        train_result['saved_path'], output_dir, verbose=verbose, plant=plant,
        trajectory_types=trajectory_types)

    # 4. 训练摘要仪表板
    p = plot_training_summary(train_result, comparison_metrics, hyperparams, output_dir,
                              scenario_labels=scenario_labels)
    if verbose:
        print(f"  训练摘要: {p}")

    # 5. 参数变化可视化
    p = plot_parameter_changes(train_result, output_dir)
    if verbose and p:
        print(f"  参数变化: {p}")

    # 6. 实验日志
    p = save_experiment_log(train_result, comparison_metrics, output_dir,
                            hyperparams, train_result['saved_path'])
    if verbose:
        print(f"  实验日志: {p}")

    # 7. 复制 tuned config 到产物目录
    shutil.copy2(train_result['saved_path'],
                 os.path.join(output_dir, os.path.basename(train_result['saved_path'])))

    if verbose:
        print(f"\n训练产物已全部保存到: {output_dir}")

    return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='独立验证：用 V1 路径跑 baseline vs tuned 对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"可用轨迹类型:\n  {', '.join(TRAJECTORY_TYPES)}")
    parser.add_argument('--config', required=True,
                        help='调参后的配置文件路径（YAML）')
    parser.add_argument('--trajectories', nargs='+', default=None,
                        help='轨迹类型名（自动展开到全速度段 + park_route）。'
                             '默认全量验证 (8×6+1=49)')
    parser.add_argument('--plant', default=None,
                        choices=['kinematic', 'dynamic', 'hybrid_dynamic', 'hybrid_v2'],
                        help='被控对象类型，默认使用配置中的值')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录，默认 results/validation/{plant}/{timestamp}/')
    args = parser.parse_args()

    run_validation(args.config, output_dir=args.output_dir, plant=args.plant,
                   trajectory_types=args.trajectories)
