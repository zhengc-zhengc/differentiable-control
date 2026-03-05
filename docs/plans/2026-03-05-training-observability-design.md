# 训练可观测性与实验自动化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 增加项目可观测性——一键体检、训练参数快照、loss 分项追踪、实验日志自动化、训练产物自动保存

**Architecture:** 新增 `health_check.py` 独立脚本 + 增强 `train.py` 的数据收集与后处理 + 新增 `optim/post_training.py` 处理训练后的可视化与日志保存。训练结果按时间戳保存到独立子目录 `results/training/{timestamp}/`。

**Tech Stack:** Python, PyTorch, matplotlib, yaml, pytest

---

### Task 1: 增强 tracking_loss 返回 loss 各加权分项

**Files:**
- Modify: `sim/optim/train.py:74-125` (tracking_loss 函数)
- Test: `sim/tests/test_train.py`

**Step 1: 修改 tracking_loss 的 details 返回值**

当前 `return_details=True` 只返回 RMSE 和 max。增加返回各加权分项的值（即 loss 的构成）：

```python
if return_details:
    details = {
        'lat_rmse': lat_mse.sqrt().item(),
        'head_rmse': head_mse.sqrt().item(),
        'speed_rmse': speed_mse.sqrt().item(),
        'lat_max': lat_errs.abs().max().item(),
        'head_max': head_errs.abs().max().item(),
        # 新增：各分项加权后的 loss 值（构成总 loss 的各部分）
        'loss_lat': (w_lat * lat_mse).item(),
        'loss_head': (w_head * head_mse).item(),
        'loss_speed': (w_speed * speed_mse).item(),
        'loss_steer_rate': (w_steer_rate * steer_rate_mse).item(),
        'loss_acc_rate': (w_acc_rate * acc_rate_mse).item(),
    }
    return loss, details
```

**Step 2: 在 test_train.py 中新增测试**

```python
def test_return_details_has_loss_components(self):
    """return_details 应包含各 loss 分项。"""
    history = [{'lateral_error': torch.tensor(0.1),
                'heading_error': torch.tensor(0.05),
                'v': torch.tensor(4.5),
                'steer': torch.tensor(float(i)),
                'acc': torch.tensor(0.1)} for i in range(20)]
    loss, details = tracking_loss(history, ref_speed=5.0, return_details=True)
    for key in ['loss_lat', 'loss_head', 'loss_speed', 'loss_steer_rate', 'loss_acc_rate']:
        assert key in details, f"缺少 {key}"
    # 各分项之和应约等于总 loss
    component_sum = sum(details[k] for k in ['loss_lat', 'loss_head', 'loss_speed',
                                              'loss_steer_rate', 'loss_acc_rate'])
    assert abs(component_sum - loss.item()) < 1e-4
```

**Step 3: 运行测试验证**

Run: `cd sim && python -m pytest tests/test_train.py -v -k "loss_components"`

**Step 4: Commit**

---

### Task 2: 增强 train() 收集完整训练历史

**Files:**
- Modify: `sim/optim/train.py:139-279` (train 函数)

**目标：** train() 内部收集每个 epoch、每条轨迹的 loss 分项和参数快照，返回给调用者。

**Step 1: 在训练循环中收集分轨迹 loss 分项**

在 train() 函数的内循环中，为每条轨迹单独记录 loss details：

```python
# 在 losses = [] 后添加
training_history = []  # 每个 epoch 的完整记录

# epoch 循环内，epoch_details 改为 per-trajectory 记录
traj_details = {}  # {traj_name: details_dict}

for traj_name in trajectories:
    # ... 现有仿真 + loss 计算 ...
    loss, details = tracking_loss(history, ref_speed=sim_speed, return_details=True)
    epoch_loss = epoch_loss + loss
    traj_details[traj_name] = details

# epoch 结束后汇总
epoch_record = {
    'epoch': epoch + 1,
    'loss': epoch_loss.item() / len(trajectories),
    'grad_norm': grad_norm,
    'nan_count': int(nan_count),
    'dt': dt,
    'per_trajectory': traj_details,
    # 各轨迹平均的 loss 分项
    'avg': {k: sum(td[k] for td in traj_details.values()) / len(trajectories)
            for k in traj_details[trajectories[0]].keys()},
}
training_history.append(epoch_record)
```

**Step 2: 每 N epoch 打印参数快照**

在 epoch 循环中，每隔 `param_snapshot_interval`（默认 10）epoch 打印参数变化：

```python
# 训练循环前记录初始参数
initial_params = {name: p.detach().clone() for name, p in params.named_parameters()}

# 在 epoch 循环中（optimizer.step() 之后）
if verbose and (epoch + 1) % param_snapshot_interval == 0:
    print(f"\n  --- 参数快照 (epoch {epoch+1}) ---")
    for name, p in params.named_parameters():
        init_val = initial_params[name]
        delta = p.detach() - init_val
        if p.numel() == 1:
            print(f"  {name:30s}: {init_val.item():.4f} -> {p.item():.4f} "
                  f"({delta.item():+.4f}, {delta.item()/max(abs(init_val.item()),1e-8)*100:+.1f}%)")
        else:
            print(f"  {name:30s}: max_delta={delta.abs().max().item():.4f} "
                  f"mean={p.detach().mean().item():.4f} [{p.detach().min().item():.3f}, {p.detach().max().item():.3f}]")
    print()
```

**Step 3: 打印分轨迹 loss 明细**

在每个 epoch 的打印中增加分轨迹行（当 verbose=True 时）：

```python
# 在现有的 epoch 打印之后
if verbose and len(trajectories) > 1:
    for tn in trajectories:
        td = traj_details[tn]
        print(f"    {tn:12s}: lat={td['lat_rmse']:.4f} head={td['head_rmse']:.4f} "
              f"spd={td['speed_rmse']:.4f} | "
              f"L_lat={td['loss_lat']:.3f} L_head={td['loss_head']:.3f} "
              f"L_spd={td['loss_speed']:.3f}")
```

**Step 4: 更新 train() 返回值**

```python
return {
    'losses': losses,
    'training_history': training_history,
    'initial_params': {name: p.cpu().tolist() if p.numel() > 1 else p.item()
                       for name, p in initial_params.items()},
    'final_params': {name: p.detach().cpu().tolist() if p.numel() > 1 else p.detach().item()
                     for name, p in params.named_parameters()},
    'saved_path': saved_path,
    'params': params,
}
```

**Step 5: 新增 `param_snapshot_interval` 参数和 CLI 参数**

在 train() 签名中加入 `param_snapshot_interval=10`。
在 argparse 中加入 `--snapshot-interval`。

**Step 6: 更新测试**

现有的 `test_pipeline_runs` 等测试需要适配新的返回值结构：

```python
def test_pipeline_runs(self):
    result = train(trajectories=['straight'], n_epochs=2,
                   lr=1e-2, sim_length=20.0, sim_speed=5.0, verbose=False)
    assert len(result['losses']) == 2
    assert len(result['training_history']) == 2
    assert 'per_trajectory' in result['training_history'][0]
    assert 'initial_params' in result
    assert 'final_params' in result
```

**Step 7: 运行全部 test_train.py 测试**

Run: `cd sim && python -m pytest tests/test_train.py -v`

**Step 8: Commit**

---

### Task 3: 新增 post_training.py — 训练后自动化

**Files:**
- Create: `sim/optim/post_training.py`
- Test: `sim/tests/test_post_training.py`

**功能：** 接收 train() 返回的数据，自动生成：
1. Loss 曲线图（总 loss + 各加权分项）
2. 分轨迹 loss 分项图
3. 调参前后对比图（轨迹 + 横向误差 + 转向角 + 加速度）
4. 实验日志 YAML

所有产物保存到 `sim/results/training/{timestamp}/` 目录。

**Step 1: 编写 post_training.py**

```python
# sim/optim/post_training.py
"""训练后自动化：生成 loss 曲线、对比图、实验日志。"""
import math
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config, _get_commit_hash, _tensor_to_python
from model.trajectory import (generate_straight, generate_circle,
                              generate_sine, generate_combined)
from sim_loop import run_simulation

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_loss_curves(training_history, output_dir):
    """绘制总 loss + 各加权分项随 epoch 变化的曲线。"""
    epochs = [r['epoch'] for r in training_history]
    total_loss = [r['loss'] for r in training_history]

    # 各分项从 avg 中取
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
    """绘制分轨迹的 loss 分项曲线（每条轨迹一个子图）。"""
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
            vals = [r['per_trajectory'][tname].get(comp, 0.0) for r in training_history]
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


_EVAL_SCENARIOS = [
    ('straight', '直线 (10 m/s)',
     lambda: generate_straight(length=200, speed=10.0), 10.0),
    ('circle', '圆弧 (R=30m, 5 m/s)',
     lambda: generate_circle(radius=30.0, speed=5.0, arc_angle=math.pi), 5.0),
    ('sine', '正弦 (A=3m, 5 m/s)',
     lambda: generate_sine(amplitude=3.0, wavelength=50.0, n_waves=2, speed=5.0), 5.0),
    ('combined', '组合 (5 m/s)',
     lambda: generate_combined(speed=5.0), 5.0),
]


def run_comparison(tuned_config_path, output_dir, verbose=True):
    """用 V1 路径（float）跑 baseline vs tuned 对比，生成对比图 + 返回指标。"""
    cfg_base = load_config()
    cfg_tuned = load_config(tuned_config_path)

    all_base = []
    all_tuned = []

    if verbose:
        header = f"{'场景':<25} {'':>8} {'lat_rmse(m)':>12} {'head_rmse(rad)':>14} {'lat_max(m)':>10}"
        print(header)
        print('-' * len(header))

    comparison_metrics = {}

    for key, name, traj_fn, init_v in _EVAL_SCENARIOS:
        traj = traj_fn()
        h_base = run_simulation(traj, init_speed=init_v, cfg=cfg_base)
        h_tuned = run_simulation(traj, init_speed=init_v, cfg=cfg_tuned)
        m_base = _calc_metrics(h_base)
        m_tuned = _calc_metrics(h_tuned)

        all_base.append((key, name, traj, h_base, m_base))
        all_tuned.append((key, name, traj, h_tuned, m_tuned))

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
            print(f"{name:<25} {'baseline':>8} {m_base['lat_rmse']:>12.4f} "
                  f"{m_base['head_rmse']:>14.4f} {m_base['lat_max']:>10.4f}")
            print(f"{'':25} {'tuned':>8} {m_tuned['lat_rmse']:>12.4f} "
                  f"{m_tuned['head_rmse']:>14.4f} {m_tuned['lat_max']:>10.4f}")
            print(f"{'':25} {'delta':>8} {d_lat:>+11.1f}% {d_head:>+13.1f}%")
            print()

    # --- 对比图 1: 轨迹 ---
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='trajectory', filename='comparison_trajectory.png')
    # --- 对比图 2: 横向误差 ---
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='lateral_error', filename='comparison_lateral_error.png')
    # --- 对比图 3: 转向角 ---
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='steer', filename='comparison_steer.png')
    # --- 对比图 4: 加速度 ---
    _plot_comparison_grid(all_base, all_tuned, output_dir,
                          plot_type='acc', filename='comparison_acc.png')

    return comparison_metrics


def _plot_comparison_grid(all_base, all_tuned, output_dir, plot_type, filename):
    """通用 2x2 对比图生成器。"""
    titles_map = {
        'trajectory': '调参前后轨迹跟踪对比',
        'lateral_error': '调参前后横向误差对比',
        'steer': '调参前后转向角输出对比',
        'acc': '调参前后加速度输出对比',
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(titles_map[plot_type], fontsize=16)

    for idx, ax in enumerate(axes.flat):
        key_b, name_b, traj_b, h_b, m_b = all_base[idx]
        _, _, _, h_t, m_t = all_tuned[idx]

        if plot_type == 'trajectory':
            ax.plot([p.x for p in traj_b], [p.y for p in traj_b],
                    'k--', label='参考轨迹', linewidth=1, alpha=0.7)
            ax.plot([h['x'] for h in h_b], [h['y'] for h in h_b],
                    'b-', label=f'调参前 (lat={m_b["lat_rmse"]:.3f}m)', linewidth=1.2, alpha=0.8)
            ax.plot([h['x'] for h in h_t], [h['y'] for h in h_t],
                    'r-', label=f'调参后 (lat={m_t["lat_rmse"]:.3f}m)', linewidth=1.2, alpha=0.8)
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
        elif plot_type == 'steer':
            ax.plot([h['t'] for h in h_b], [h['steer'] for h in h_b],
                    'b-', label='调参前', alpha=0.8)
            ax.plot([h['t'] for h in h_t], [h['steer'] for h in h_t],
                    'r-', label='调参后', alpha=0.8)
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

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_experiment_log(train_result, comparison_metrics, output_dir,
                        hyperparams, tuned_config_path):
    """保存实验日志 YAML。"""
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
        'comparison': comparison_metrics,
        'parameter_changes': {},
    }

    # 参数变化
    for name in train_result['initial_params']:
        init_v = train_result['initial_params'][name]
        final_v = train_result['final_params'][name]
        if isinstance(init_v, (int, float)):
            delta = final_v - init_v
            pct = delta / max(abs(init_v), 1e-8) * 100
            log['parameter_changes'][name] = {
                'initial': round(init_v, 6),
                'final': round(final_v, 6),
                'delta': round(delta, 6),
                'delta_pct': round(pct, 2),
            }
        else:
            # 查找表：只记录最大变化
            max_delta = max(abs(f - i) for f, i in zip(final_v, init_v))
            log['parameter_changes'][name] = {
                'max_delta': round(max_delta, 6),
                'final_range': [round(min(final_v), 4), round(max(final_v), 4)],
            }

    path = os.path.join(output_dir, 'experiment_log.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(_tensor_to_python(log), f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    return path


def run_post_training(train_result, hyperparams, verbose=True):
    """训练后一站式自动化入口。

    Args:
        train_result: train() 的返回值
        hyperparams: 训练超参数 dict
        verbose: 是否打印进度

    Returns:
        output_dir: 产物保存目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sim_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = _ensure_dir(os.path.join(sim_dir, 'results', 'training', timestamp))

    if verbose:
        print(f"\n{'='*60}")
        print(f"训练后自动化 — 产物保存到: {output_dir}")
        print(f"{'='*60}")

    # 1. Loss 曲线
    p = plot_loss_curves(train_result['training_history'], output_dir)
    if verbose:
        print(f"  Loss 曲线: {p}")

    # 2. 分轨迹 loss 分项
    trajectories = hyperparams.get('trajectories', ['circle', 'sine', 'combined'])
    p = plot_loss_breakdown(train_result['training_history'], trajectories, output_dir)
    if verbose:
        print(f"  Loss 分项: {p}")

    # 3. V1 路径对比
    if verbose:
        print(f"\n  --- V1 路径验证（baseline vs tuned）---")
    comparison_metrics = run_comparison(train_result['saved_path'], output_dir,
                                        verbose=verbose)

    # 4. 实验日志
    p = save_experiment_log(train_result, comparison_metrics, output_dir,
                            hyperparams, train_result['saved_path'])
    if verbose:
        print(f"  实验日志: {p}")

    # 5. 复制 tuned config 到产物目录（方便一并归档）
    import shutil
    shutil.copy2(train_result['saved_path'],
                 os.path.join(output_dir, os.path.basename(train_result['saved_path'])))

    if verbose:
        print(f"\n训练产物已全部保存到: {output_dir}")

    return output_dir
```

**Step 2: 编写测试 test_post_training.py**

```python
# sim/tests/test_post_training.py
"""训练后自动化测试。"""
import os
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPostTraining:
    def _make_mock_train_result(self):
        """构造模拟的 train() 返回值，避免真正跑训练。"""
        return {
            'losses': [3.66, 3.62, 3.60],
            'training_history': [
                {
                    'epoch': i + 1,
                    'loss': 3.66 - i * 0.03,
                    'grad_norm': 4.0,
                    'nan_count': 0,
                    'dt': 10.0,
                    'per_trajectory': {
                        'circle': {
                            'lat_rmse': 0.085, 'head_rmse': 0.023,
                            'speed_rmse': 0.003, 'lat_max': 0.15, 'head_max': 0.05,
                            'loss_lat': 0.72, 'loss_head': 0.26,
                            'loss_speed': 0.009, 'loss_steer_rate': 0.001,
                            'loss_acc_rate': 0.001,
                        },
                        'sine': {
                            'lat_rmse': 0.152, 'head_rmse': 0.041,
                            'speed_rmse': 0.005, 'lat_max': 0.25, 'head_max': 0.08,
                            'loss_lat': 2.31, 'loss_head': 0.84,
                            'loss_speed': 0.025, 'loss_steer_rate': 0.002,
                            'loss_acc_rate': 0.001,
                        },
                    },
                    'avg': {
                        'lat_rmse': 0.118, 'head_rmse': 0.032,
                        'speed_rmse': 0.004, 'lat_max': 0.20, 'head_max': 0.065,
                        'loss_lat': 1.51, 'loss_head': 0.55,
                        'loss_speed': 0.017, 'loss_steer_rate': 0.0015,
                        'loss_acc_rate': 0.001,
                    },
                }
                for i in range(3)
            ],
            'initial_params': {
                'lon_ctrl.station_kp': 0.25,
                'lon_ctrl.station_ki': 0.0,
                'lat_ctrl.T2_y': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            },
            'final_params': {
                'lon_ctrl.station_kp': 0.263,
                'lon_ctrl.station_ki': 0.001,
                'lat_ctrl.T2_y': [1.51, 1.49, 1.52, 1.50, 1.48, 1.51, 1.50],
            },
            'saved_path': None,  # 需要在测试中设置为真实路径
            'params': None,
        }

    def test_plot_loss_curves(self, tmp_path):
        from optim.post_training import plot_loss_curves
        result = self._make_mock_train_result()
        path = plot_loss_curves(result['training_history'], str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('loss_curve.png')

    def test_plot_loss_breakdown(self, tmp_path):
        from optim.post_training import plot_loss_breakdown
        result = self._make_mock_train_result()
        path = plot_loss_breakdown(result['training_history'],
                                   ['circle', 'sine'], str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('loss_breakdown.png')

    def test_save_experiment_log(self, tmp_path):
        import yaml
        from optim.post_training import save_experiment_log
        result = self._make_mock_train_result()
        # 创建一个临时的 tuned config 用于测试
        from config import load_config, save_tuned_config
        cfg = load_config()
        saved = save_tuned_config(cfg, output_dir=str(tmp_path))
        result['saved_path'] = saved

        hyperparams = {'epochs': 3, 'lr': 1e-3, 'trajectories': ['circle', 'sine']}
        path = save_experiment_log(result, {}, str(tmp_path), hyperparams, saved)
        assert os.path.exists(path)
        with open(path, 'r') as f:
            log = yaml.safe_load(f)
        assert 'commit' in log
        assert 'hyperparams' in log
        assert log['results']['initial_loss'] == pytest.approx(3.66)
```

**Step 3: 运行测试**

Run: `cd sim && python -m pytest tests/test_post_training.py -v`

**Step 4: Commit**

---

### Task 4: 在 train.py 的 __main__ 中集成 post_training

**Files:**
- Modify: `sim/optim/train.py:282-308` (__main__ 块)

**Step 1: 在 __main__ 中调用 run_post_training**

```python
if __name__ == '__main__':
    # ... 现有 argparse ...
    args = parser.parse_args()

    result = train(trajectories=args.trajectories, n_epochs=args.epochs,
                   lr=args.lr, sim_speed=args.speed,
                   sim_length=args.sim_length,
                   tbptt_k=args.tbptt_k,
                   grad_clip=args.grad_clip)
    print(f"\n最终 loss: {result['losses'][-1]:.6f}")
    print(f"保存路径: {result['saved_path']}")

    # 训练后自动化
    from optim.post_training import run_post_training
    hyperparams = {
        'epochs': args.epochs,
        'lr': args.lr,
        'trajectories': args.trajectories,
        'speed': args.speed,
        'sim_length': args.sim_length,
        'tbptt_k': args.tbptt_k,
        'grad_clip': args.grad_clip,
    }
    run_post_training(result, hyperparams)
```

**Step 2: 运行 5 epoch 短训练进行端到端验证**

Run: `cd sim && python optim/train.py --epochs 5 --trajectories circle sine --sim-length 30 --speed 5.0`

验证：
- 每 epoch 打印 loss + 分轨迹明细
- 训练完成后自动生成 loss 曲线、对比图、实验日志
- 检查 `results/training/{timestamp}/` 目录内容完整

**Step 3: Commit**

---

### Task 5: 新增 health_check.py — 一键体检

**Files:**
- Create: `sim/health_check.py`
- Test: `sim/tests/test_health_check.py`

**Step 1: 编写 health_check.py**

```python
# sim/health_check.py
"""一键项目体检：测试 + 基线性能 + 梯度健康检查。"""
import math
import os
import sys
import subprocess
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from model.trajectory import (generate_straight, generate_circle,
                              generate_sine, generate_combined)
from sim_loop import run_simulation
from optim.train import DiffControllerParams, tracking_loss, _TRAJECTORY_BUILDERS


def run_pytest():
    """运行 pytest，返回 (passed, failed, total)。"""
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_dir, '-q', '--tb=no'],
        capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    # 解析 "X passed, Y failed" 格式
    passed = failed = 0
    for line in output.split('\n'):
        if 'passed' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'passed' or p == 'passed,':
                    passed = int(parts[i - 1])
                if p == 'failed' or p == 'failed,':
                    failed = int(parts[i - 1])
    return passed, failed, result.returncode


def check_baseline_performance():
    """用 default.yaml 跑 4 场景，返回指标。"""
    scenarios = [
        ('直线 (10 m/s)', generate_straight(length=200, speed=10.0), 10.0),
        ('圆弧 (R=30m, 5 m/s)', generate_circle(radius=30.0, speed=5.0, arc_angle=math.pi), 5.0),
        ('正弦 (A=3m, 5 m/s)', generate_sine(amplitude=3.0, wavelength=50.0, n_waves=2, speed=5.0), 5.0),
        ('组合 (5 m/s)', generate_combined(speed=5.0), 5.0),
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
    """跑 1 epoch 训练，检查梯度健康状况。返回参数级梯度信息。"""
    if trajectories is None:
        trajectories = ['circle', 'sine', 'combined']

    params = DiffControllerParams()
    # 跑 1 epoch
    epoch_loss = torch.tensor(0.0)
    for traj_name in trajectories:
        builder = _TRAJECTORY_BUILDERS[traj_name]
        traj = builder(sim_speed)
        history = run_simulation(
            traj, init_speed=sim_speed, cfg=params.cfg,
            lat_ctrl=params.lat_ctrl, lon_ctrl=params.lon_ctrl,
            differentiable=True, tbptt_k=tbptt_k)
        loss = tracking_loss(history, ref_speed=sim_speed)
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

        # 状态判定
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
        print(f"  {r['name']:<25} {r['lat_rmse']:>12.4f} {r['head_rmse']:>14.4f} {r['lat_max']:>10.4f}")
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
            mark = '[ERROR]'
        print(f"  {g['name']:<35} {g['value']:>18} {g['grad']:>18} {mark:>8}")

    # 总结
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
```

**Step 2: 编写测试 test_health_check.py**

```python
# sim/tests/test_health_check.py
"""health_check 模块测试。"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestHealthCheck:
    def test_check_baseline_performance(self):
        from health_check import check_baseline_performance
        results = check_baseline_performance()
        assert len(results) == 4
        for r in results:
            assert 'lat_rmse' in r
            assert r['lat_rmse'] >= 0
            assert r['lat_rmse'] < 1.0  # 不应超过 1m

    def test_check_gradient_health(self):
        from health_check import check_gradient_health
        grad_info, total_norm, loss = check_gradient_health(
            trajectories=['circle'], sim_speed=5.0)
        assert len(grad_info) > 0
        assert total_norm >= 0
        assert loss > 0
        # 至少部分参数应有梯度
        n_ok = sum(1 for g in grad_info if g['status'] == 'OK')
        assert n_ok > 0, "应至少有部分参数有非零梯度"

    def test_gradient_info_fields(self):
        from health_check import check_gradient_health
        grad_info, _, _ = check_gradient_health(
            trajectories=['circle'], sim_speed=5.0)
        for g in grad_info:
            assert 'name' in g
            assert 'status' in g
            assert g['status'] in ('OK', 'WARN_ZERO', 'WARN_SMALL', 'ERROR')
```

**Step 3: 运行测试**

Run: `cd sim && python -m pytest tests/test_health_check.py -v`

**Step 4: 运行 health_check 端到端验证**

Run: `cd sim && python health_check.py`

验证输出包含三个部分：测试结果、基线性能表格、梯度健康表格。

**Step 5: Commit**

---

### Task 6: 更新 CLAUDE.md + 添加 experiments/ 到 .gitignore

**Files:**
- Modify: `sim/CLAUDE.md`
- Modify: `.gitignore`（如有，添加 `sim/results/training/` 和 `sim/experiments/` 的排除规则，视需要决定）

**Step 1: 更新 sim/CLAUDE.md 的模块结构和常用命令**

在模块结构中加入：
```
├── health_check.py        # 一键体检（测试 + 基线性能 + 梯度健康）
├── optim/
│   ├── train.py           # 训练（增强：参数快照、分轨迹 loss、完整训练历史）
│   └── post_training.py   # 训练后自动化（loss 曲线、对比图、实验日志）
├── results/
│   ├── training/          # 训练产物（每次训练一个时间戳子目录）
│   │   └── {timestamp}/
│   │       ├── loss_curve.png
│   │       ├── loss_breakdown.png
│   │       ├── comparison_trajectory.png
│   │       ├── comparison_lateral_error.png
│   │       ├── comparison_steer.png
│   │       ├── comparison_acc.png
│   │       └── experiment_log.yaml
```

在常用命令中加入：
```bash
python health_check.py                                      # 一键体检
```

**Step 2: Commit**

---

### Task 7: 全量测试 + 端到端验证

**Step 1: 运行全部测试**

Run: `cd sim && python -m pytest tests/ -v`

Expected: 所有测试通过（包括新增的 test_post_training.py 和 test_health_check.py）

**Step 2: 运行 health_check**

Run: `cd sim && python health_check.py`

**Step 3: 运行短训练端到端**

Run: `cd sim && python optim/train.py --epochs 5 --trajectories circle sine --speed 5.0`

验证 `results/training/{timestamp}/` 下有完整产物。

**Step 4: Final commit + push**
