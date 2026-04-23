"""49 场景对比：truck_trailer vs hybrid_dynamic。

同一套控制器参数（configs/default.yaml，yawrate 走 plant 真值）、同一批 49 条
轨迹、两种 plant，batched V1 跑一遍，比较 lat/head RMSE。
"""
from __future__ import annotations

import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'sim'))
sys.path.insert(0, SIM_DIR)

from config import apply_plant_override, load_config
from optim.post_training import _build_eval_scenarios, _calc_metrics
from optim.train_batch import run_simulation_batch
from optim.validate_batch import _batched_to_scenario_histories


def run_plant(plant_name, scenarios, trajs):
    cfg = load_config(None)
    apply_plant_override(cfg, plant_name)
    dt = cfg['simulation']['dt']
    t0 = time.time()
    batch = run_simulation_batch(trajs, cfg=cfg, tbptt_k=0, hard_mode=True)
    elapsed = time.time() - t0
    hists = _batched_to_scenario_histories(batch, len(trajs), dt)
    metrics = [_calc_metrics(h) for h in hists]
    return metrics, elapsed


def main():
    scenarios = _build_eval_scenarios(None)
    trajs = [gen() for _k, _n, gen in scenarios]
    n = len(trajs)
    print(f"场景数: {n}\n")

    print("[1/2] truck_trailer plant ...")
    m_tr, t_tr = run_plant('truck_trailer', scenarios, trajs)
    print(f"  完成 {t_tr:.1f}s")

    print("[2/2] hybrid_dynamic plant ...")
    m_hy, t_hy = run_plant('hybrid_dynamic', scenarios, trajs)
    print(f"  完成 {t_hy:.1f}s\n")

    print(f"{'场景':<32} {'truck lat':>10} {'hybrid lat':>10} "
          f"{'Δ':>8}  {'truck head':>11} {'hybrid head':>11} {'Δ':>8}")
    print('-' * 96)
    rows = {}
    tr_sum_lat = hy_sum_lat = tr_sum_head = hy_sum_head = 0.0
    tr_max = hy_max = 0.0
    for (key, name, _g), a, b in zip(scenarios, m_tr, m_hy):
        d_lat = b['lat_rmse'] - a['lat_rmse']
        d_head = b['head_rmse'] - a['head_rmse']
        tr_sum_lat += a['lat_rmse']; hy_sum_lat += b['lat_rmse']
        tr_sum_head += a['head_rmse']; hy_sum_head += b['head_rmse']
        tr_max = max(tr_max, a['lat_max']); hy_max = max(hy_max, b['lat_max'])
        rows[key] = {
            'name': name,
            'truck_lat_rmse':   round(a['lat_rmse'], 4),
            'hybrid_lat_rmse':  round(b['lat_rmse'], 4),
            'delta_lat':        round(d_lat, 4),
            'truck_head_rmse':  round(a['head_rmse'], 4),
            'hybrid_head_rmse': round(b['head_rmse'], 4),
            'delta_head':       round(d_head, 4),
            'truck_lat_max':    round(a['lat_max'], 4),
            'hybrid_lat_max':   round(b['lat_max'], 4),
        }
        print(f"{name:<32} {a['lat_rmse']:10.4f} {b['lat_rmse']:10.4f} "
              f"{d_lat:+8.3f}  {a['head_rmse']:11.4f} {b['head_rmse']:11.4f} "
              f"{d_head:+8.4f}")
    print('-' * 96)
    print(f"{'平均':<32} {tr_sum_lat/n:10.4f} {hy_sum_lat/n:10.4f} "
          f"{(hy_sum_lat-tr_sum_lat)/n:+8.3f}  "
          f"{tr_sum_head/n:11.4f} {hy_sum_head/n:11.4f} "
          f"{(hy_sum_head-tr_sum_head)/n:+8.4f}")
    print(f"{'最大 lat_max':<32} {tr_max:10.4f} {hy_max:10.4f}")
    print()

    # 按场景比较：谁好、改善多少
    better_hybrid = sum(1 for r in rows.values()
                       if r['hybrid_lat_rmse'] < r['truck_lat_rmse'])
    tie_lat = sum(1 for r in rows.values()
                   if r['hybrid_lat_rmse'] == r['truck_lat_rmse'])
    print(f"横向：hybrid_dynamic 优于 truck_trailer 的场景数 = "
          f"{better_hybrid}/{n}，持平 {tie_lat}")
    # hybrid 相对 truck 的平均变化幅度
    ratios = []
    for r in rows.values():
        if r['truck_lat_rmse'] > 1e-6:
            ratios.append((r['hybrid_lat_rmse'] - r['truck_lat_rmse']) /
                           r['truck_lat_rmse'] * 100)
    if ratios:
        print(f"横向 RMSE 平均变化：{sum(ratios)/len(ratios):+.2f}%，"
              f"最大改善 {min(ratios):+.1f}%，最大恶化 {max(ratios):+.1f}%")

    output_dir = os.path.join(
        SIM_DIR, 'results', 'validation',
        time.strftime('%Y%m%d_%H%M%S') + '_plant_compare')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'plant_compare.yaml'), 'w',
              encoding='utf-8') as f:
        yaml.safe_dump({
            'n_scenarios': n,
            'sim_time_truck_s': round(t_tr, 1),
            'sim_time_hybrid_s': round(t_hy, 1),
            'truck_trailer_avg_lat_rmse':   round(tr_sum_lat/n, 4),
            'hybrid_dynamic_avg_lat_rmse':  round(hy_sum_lat/n, 4),
            'truck_trailer_avg_head_rmse':  round(tr_sum_head/n, 4),
            'hybrid_dynamic_avg_head_rmse': round(hy_sum_head/n, 4),
            'per_scenario': rows,
        }, f, allow_unicode=True, sort_keys=False)
    print(f"\n输出: {output_dir}")


if __name__ == '__main__':
    main()
