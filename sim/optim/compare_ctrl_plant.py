"""2×2 控制器-被控对象对比：
  控制器 T = 当前卡车控制器 (configs/default.yaml)
  控制器 R = 之前 roboVAN 控制器 (configs/default_robovan.yaml)
  被控对象 P1 = truck_trailer
  被控对象 P2 = hybrid_v2

对每个被控对象，都让两种控制器跑 49 条验证轨迹，并绘制对比图
（trajectory / lateral_error / speed_error / steer / acc）。
图风格对齐 sim/results 中已有的 comparison_*.png。

产物: sim/results/ctrl_plant_comparison/<plant>/
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import apply_plant_override, load_config
from sim_loop import run_simulation
from optim.post_training import _build_eval_scenarios, _calc_metrics
from optim.validate_batch import (_batched_to_scenario_histories,
                                   _plot_comparison_grid_custom)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_TRUCK = os.path.join(SIM_DIR, 'configs', 'default.yaml')
CFG_ROBOVAN = os.path.join(SIM_DIR, 'configs', 'default_robovan.yaml')


def _run_scalar_all_scenarios(cfg, scenarios, label):
    """scalar 路径跑完所有 49 场景，返回 all_* 列表。"""
    out = []
    n = len(scenarios)
    t0 = time.time()
    for i, (key, name, gen) in enumerate(scenarios):
        traj = gen()
        init_v = float(traj[0].v)
        h = run_simulation(traj, init_speed=init_v, cfg=cfg)
        m = _calc_metrics(h)
        out.append((key, name, traj, h, m, init_v))
        print(f"  [{label}] [{i+1:2d}/{n}] {name:<30} "
              f"lat_rmse={m['lat_rmse']:.4f}  head_rmse={m['head_rmse']:.4f}")
    print(f"  [{label}] 完成 {n} 场景，共 {time.time()-t0:.1f}s\n")
    return out


def _run_batched_truck_trailer(cfg, scenarios, label):
    from optim.train_batch import run_simulation_batch
    trajs = [gen() for _k, _lbl, gen in scenarios]
    init_vs = [float(t[0].v) for t in trajs]
    dt = cfg['simulation']['dt']
    t0 = time.time()
    hist_b = run_simulation_batch(trajs, cfg=cfg, tbptt_k=0, hard_mode=True)
    hists = _batched_to_scenario_histories(hist_b, len(trajs), dt)
    print(f"  [{label}] 批量仿真完成，共 {time.time()-t0:.1f}s")
    out = []
    for (key, name, _gen), traj, h, init_v in zip(scenarios, trajs, hists, init_vs):
        m = _calc_metrics(h)
        out.append((key, name, traj, h, m, init_v))
    return out


def run_plant_comparison(plant, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scenarios = _build_eval_scenarios(None)
    print(f"\n{'='*64}\n被控对象: {plant}    场景数: {len(scenarios)}    输出: {out_dir}\n{'='*64}")

    cfg_t = load_config(CFG_TRUCK); apply_plant_override(cfg_t, plant)
    cfg_r = load_config(CFG_ROBOVAN); apply_plant_override(cfg_r, plant)

    if plant == 'truck_trailer':
        all_t = _run_batched_truck_trailer(cfg_t, scenarios, '卡车控制器')
        all_r = _run_batched_truck_trailer(cfg_r, scenarios, 'roboVAN控制器')
    else:
        all_t = _run_scalar_all_scenarios(cfg_t, scenarios, '卡车控制器')
        all_r = _run_scalar_all_scenarios(cfg_r, scenarios, 'roboVAN控制器')

    # metrics 汇总
    comparison = {}
    for (key, name, _tr, _h, m_t, _v), (_k2, _n2, _tr2, _h2, m_r, _v2) in zip(all_t, all_r):
        d_lat = (m_r['lat_rmse'] - m_t['lat_rmse']) / m_t['lat_rmse'] * 100 if m_t['lat_rmse'] > 1e-8 else 0.0
        d_head = (m_r['head_rmse'] - m_t['head_rmse']) / m_t['head_rmse'] * 100 if m_t['head_rmse'] > 1e-8 else 0.0
        comparison[key] = {
            'name': name,
            'truck_lat_rmse': round(m_t['lat_rmse'], 5),
            'robovan_lat_rmse': round(m_r['lat_rmse'], 5),
            'delta_lat_pct': round(d_lat, 2),
            'truck_head_rmse': round(m_t['head_rmse'], 5),
            'robovan_head_rmse': round(m_r['head_rmse'], 5),
            'delta_head_pct': round(d_head, 2),
            'truck_lat_max': round(m_t['lat_max'], 5),
            'robovan_lat_max': round(m_r['lat_max'], 5),
        }

    avg_lat_t = sum(v['truck_lat_rmse'] for v in comparison.values()) / len(comparison)
    avg_lat_r = sum(v['robovan_lat_rmse'] for v in comparison.values()) / len(comparison)
    avg_head_t = sum(v['truck_head_rmse'] for v in comparison.values()) / len(comparison)
    avg_head_r = sum(v['robovan_head_rmse'] for v in comparison.values()) / len(comparison)
    summary = {
        'plant': plant,
        'n_scenarios': len(comparison),
        'avg_truck_lat_rmse': round(avg_lat_t, 5),
        'avg_robovan_lat_rmse': round(avg_lat_r, 5),
        'avg_truck_head_rmse': round(avg_head_t, 5),
        'avg_robovan_head_rmse': round(avg_head_r, 5),
        'per_scenario': comparison,
    }
    with open(os.path.join(out_dir, 'summary.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    title_prefix = f'[{plant}] 卡车 vs roboVAN 控制器 '
    for plot_type, fname in [
            ('trajectory', 'comparison_trajectory.png'),
            ('lateral_error', 'comparison_lateral_error.png'),
            ('speed_error', 'comparison_speed_error.png'),
            ('steer', 'comparison_steer.png'),
            ('acc', 'comparison_acc.png')]:
        _plot_comparison_grid_custom(
            all_t, all_r, out_dir, plot_type, fname,
            label_a='卡车控制器', label_b='roboVAN控制器',
            title_prefix=title_prefix)

    print(f"\n[{plant}] 汇总:")
    print(f"  平均 lat_rmse:  卡车={avg_lat_t:.4f}  roboVAN={avg_lat_r:.4f}")
    print(f"  平均 head_rmse: 卡车={avg_head_t:.4f}  roboVAN={avg_head_r:.4f}")
    return summary


def plot_cross_summary(summaries, out_dir):
    """四格柱状图：2×2 控制器-被控对象的平均 lat_rmse / head_rmse。"""
    import numpy as np
    plants = list(summaries.keys())
    ctrls = ['truck', 'robovan']
    ctrl_labels = ['卡车控制器', 'roboVAN控制器']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('2×2 控制器-被控对象对比（49 场景平均 RMSE）', fontsize=14)

    for ax, metric, ylabel in zip(
            axes, ['lat_rmse', 'head_rmse'],
            ['平均横向误差 RMSE (m)', '平均航向误差 RMSE (rad)']):
        x = np.arange(len(plants))
        width = 0.35
        for i, ctrl in enumerate(ctrls):
            vals = [summaries[p][f'avg_{ctrl}_{metric}'] for p in plants]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=ctrl_labels[i])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(plants)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'cross_summary.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    root = os.path.join(SIM_DIR, 'results', 'ctrl_plant_comparison')
    os.makedirs(root, exist_ok=True)

    summaries = {}
    for plant in ['truck_trailer', 'hybrid_v2']:
        out = os.path.join(root, plant)
        summaries[plant] = run_plant_comparison(plant, out)

    p = plot_cross_summary(summaries, root)
    print(f"\n跨 plant 汇总图: {p}")

    with open(os.path.join(root, 'cross_summary.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(summaries, f, allow_unicode=True, sort_keys=False)
    print(f"所有产物保存于: {root}")


if __name__ == '__main__':
    main()
