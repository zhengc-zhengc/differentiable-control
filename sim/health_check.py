# sim/health_check.py
"""一键项目体检：测试 + 基线性能 + 梯度健康检查。

用法：
    cd sim/
    python health_check.py
"""
import math
import os
import sys
import subprocess
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from model.trajectory import (generate_straight, generate_circle,
                              generate_combined,
                              generate_double_lane_change)
from sim_loop import run_simulation
from optim.train import DiffControllerParams, tracking_loss, _TRAJECTORY_BUILDERS


def run_pytest():
    """运行 pytest，返回 (passed, failed, returncode)。"""
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_dir, '-q', '--tb=no'],
        capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    passed = failed = 0
    for line in output.split('\n'):
        if 'passed' in line or 'failed' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if 'passed' in p and i > 0:
                    try:
                        passed = int(parts[i - 1])
                    except ValueError:
                        pass
                if 'failed' in p and i > 0:
                    try:
                        failed = int(parts[i - 1])
                    except ValueError:
                        pass
    return passed, failed, result.returncode


def check_baseline_performance():
    """用 default.yaml 跑 4 场景，返回指标列表。"""
    scenarios = [
        ('直线 (10 m/s)',
         generate_straight(length=200, speed=10.0), 10.0),
        ('圆弧 (R=30m, 5 m/s)',
         generate_circle(radius=30.0, speed=5.0, arc_angle=math.pi), 5.0),
        ('双换道 (5 m/s)',
         generate_double_lane_change(lane_width=3.5, change_length=50.0, speed=5.0), 5.0),
        ('组合 (5 m/s)',
         generate_combined(speed=5.0), 5.0),
    ]
    results = []
    for name, traj, init_v in scenarios:
        history = run_simulation(traj, init_speed=init_v)
        lat = [abs(h['lateral_error']) for h in history]
        head = [abs(h['heading_error']) for h in history]
        lat_rmse = (sum(e**2 for e in lat) / len(lat)) ** 0.5
        head_rmse = (sum(e**2 for e in head) / len(head)) ** 0.5
        results.append({
            'name': name,
            'lat_rmse': lat_rmse,
            'head_rmse': head_rmse,
            'lat_max': max(lat),
        })
    return results


def check_gradient_health(trajectories=None, sim_speed=5.0, tbptt_k=64):
    """跑 1 epoch 训练（不更新参数），检查所有参数的梯度健康。

    Returns:
        grad_info: list of dicts，每个参数的梯度信息
        total_norm: float，所有参数的梯度总范数
        loss: float，1 epoch 的 loss 值
    """
    if trajectories is None:
        trajectories = ['circle', 'combined', 'double_lane_change']

    params = DiffControllerParams()

    epoch_loss = torch.tensor(0.0)
    for traj_name in trajectories:
        builder = _TRAJECTORY_BUILDERS[traj_name]
        traj = builder(sim_speed)
        traj_speed = traj[0].v
        history = run_simulation(
            traj, init_speed=traj_speed, cfg=params.cfg,
            lat_ctrl=params.lat_ctrl, lon_ctrl=params.lon_ctrl,
            differentiable=True, tbptt_k=tbptt_k)
        loss = tracking_loss(history, ref_speed=traj_speed)
        epoch_loss = epoch_loss + loss

    epoch_loss = epoch_loss / len(trajectories)
    epoch_loss.backward()

    grad_info = []
    total_norm_sq = 0.0
    for name, p in params.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            has_nan = torch.isnan(p.grad).any().item()
            has_inf = torch.isinf(p.grad).any().item()
        else:
            grad_norm = 0.0
            has_nan = False
            has_inf = False

        total_norm_sq += grad_norm ** 2

        if p.numel() == 1:
            val_str = f"{p.item():.4f}"
            grad_str = f"{p.grad.item():.6f}" if p.grad is not None else "None"
        else:
            val_str = f"[{p.detach().min().item():.3f}, {p.detach().max().item():.3f}]"
            grad_str = f"norm={grad_norm:.6f}" if p.grad is not None else "None"

        if has_nan or has_inf:
            status = 'ERROR'
        elif grad_norm < 1e-10:
            status = 'WARN_ZERO'
        elif grad_norm < 1e-6:
            status = 'WARN_SMALL'
        else:
            status = 'OK'

        grad_info.append({
            'name': name,
            'numel': p.numel(),
            'value': val_str,
            'grad': grad_str,
            'grad_norm': grad_norm,
            'status': status,
            'has_nan': has_nan,
            'has_inf': has_inf,
        })

    total_norm = total_norm_sq ** 0.5
    return grad_info, total_norm, epoch_loss.item()


def main():
    """运行完整体检，打印报告。"""
    print("=" * 65)
    print("  项目体检报告")
    print("=" * 65)

    # 1. 测试
    print("\n[1/3] 运行测试...")
    t0 = time.time()
    passed, failed, rc = run_pytest()
    dt = time.time() - t0
    total = passed + failed
    status = "PASS" if rc == 0 else "FAIL"
    print(f"  {passed}/{total} 通过  ({dt:.1f}s)  [{status}]")
    if failed > 0:
        print(f"  !! {failed} 个测试失败 !!")

    # 2. 基线性能
    print("\n[2/3] 基线性能 (default.yaml)...")
    t0 = time.time()
    perf = check_baseline_performance()
    dt = time.time() - t0
    print(f"  {'场景':<25} {'lat_rmse(m)':>12} {'head_rmse(rad)':>14} {'lat_max(m)':>10}")
    print(f"  {'-'*61}")
    for r in perf:
        print(f"  {r['name']:<25} {r['lat_rmse']:>12.4f} {r['head_rmse']:>14.4f} "
              f"{r['lat_max']:>10.4f}")
    print(f"  ({dt:.1f}s)")

    # 3. 梯度健康
    print("\n[3/3] 梯度健康检查 (1 epoch)...")
    t0 = time.time()
    grad_info, total_norm, loss = check_gradient_health()
    dt = time.time() - t0
    print(f"  Loss: {loss:.4f}  |  总梯度范数: {total_norm:.4f}  ({dt:.1f}s)")
    print()
    print(f"  {'参数':<35} {'值':>18} {'梯度':>18} {'状态':>8}")
    print(f"  {'-'*79}")
    for g in grad_info:
        if g['status'] == 'OK':
            mark = '[OK]'
        elif g['status'] == 'WARN_ZERO':
            mark = '[!!零]'
        elif g['status'] == 'WARN_SMALL':
            mark = '[!小]'
        else:
            mark = '[ERR]'
        print(f"  {g['name']:<35} {g['value']:>18} {g['grad']:>18} {mark:>8}")

    n_ok = sum(1 for g in grad_info if g['status'] == 'OK')
    n_warn = sum(1 for g in grad_info if g['status'].startswith('WARN'))
    n_err = sum(1 for g in grad_info if g['status'] == 'ERROR')
    print(f"\n  梯度汇总: {n_ok} 正常, {n_warn} 警告, {n_err} 错误")

    print("\n" + "=" * 65)
    all_ok = (rc == 0 and n_err == 0)
    print(f"  体检结论: {'全部正常' if all_ok else '存在问题，请检查上方详情'}")
    print("=" * 65)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
