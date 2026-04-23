"""跑一次 49 场景 V1 验证，打印各场景横向/航向 RMSE。
用于确认改成 plant 真实 yawrate 之后，49 场景数字与之前 A/B 实验里的 B 一致。
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'sim'))
sys.path.insert(0, SIM_DIR)

from config import load_config, apply_plant_override
from optim.post_training import _build_eval_scenarios, _calc_metrics
from optim.train_batch import run_simulation_batch
from optim.validate_batch import _batched_to_scenario_histories


def main():
    cfg = load_config(None)
    apply_plant_override(cfg, 'truck_trailer')
    dt = cfg['simulation']['dt']

    scenarios = _build_eval_scenarios(None)
    trajs = [gen() for _k, _n, gen in scenarios]
    n = len(trajs)

    t0 = time.time()
    batch_hist = run_simulation_batch(trajs, cfg=cfg, tbptt_k=0,
                                      hard_mode=True)
    print(f"batch 仿真：{time.time() - t0:.1f}s  场景数={n}\n")

    hists = _batched_to_scenario_histories(batch_hist, n, dt)

    print(f"{'场景':<32} {'lat RMSE(m)':>12} {'head RMSE(rad)':>15} "
          f"{'lat max(m)':>12}")
    print('-' * 76)
    sum_lat = sum_head = 0.0
    for (key, name, _g), h in zip(scenarios, hists):
        m = _calc_metrics(h)
        sum_lat += m['lat_rmse']
        sum_head += m['head_rmse']
        print(f"{name:<32} {m['lat_rmse']:12.4f} {m['head_rmse']:15.4f} "
              f"{m['lat_max']:12.4f}")
    print('-' * 76)
    print(f"{'平均':<32} {sum_lat/n:12.4f} {sum_head/n:15.4f}")


if __name__ == '__main__':
    main()
