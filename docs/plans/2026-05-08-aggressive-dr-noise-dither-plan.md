# 激进 DR：噪声 + 抖动 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保守档物理参数 DR 之上，给 `train_batch.py` 加状态反馈高斯白噪声 + 指令高频抖动，让控制器对感知/执行不确定性也鲁棒。

**Architecture:** 噪声采样独立工具函数（`sim/common.py`），训练时由 yaml + CLI 合并配置后构造 `torch.Generator` 锁种子，注入位置在 `run_simulation_batch` 内部循环——状态噪声加在 vehicle 状态进控制器之前，指令抖动加在控制器输出送 `vehicle.step` 之前。loss 仍用 vehicle 真值，hard_mode 自动关噪。

**Tech Stack:** PyTorch（已用），argparse，pytest（已用）。所有改动控制在 `sim/optim/train_batch.py / sim/common.py / sim/configs/default.yaml / sim/tests/`，零新增依赖。

**Spec:** [`docs/plans/2026-05-08-aggressive-dr-noise-dither-design.md`](2026-05-08-aggressive-dr-noise-dither-design.md)

---

## Task 1: 噪声采样工具函数 `sample_clipped_normal`

**Files:**
- Modify: `sim/common.py` （末尾追加函数）
- Modify: `sim/tests/test_common.py` （末尾追加测试类 `TestSampleClippedNormal`）

- [ ] **Step 1: 写失败测试**

把以下加到 `sim/tests/test_common.py` 末尾：

```python
class TestSampleClippedNormal:
    def test_shape_and_dtype(self):
        from common import sample_clipped_normal
        out = sample_clipped_normal(B=192, sigma=0.05, generator=None)
        assert out.shape == (192,)
        assert out.dtype == torch.float32

    def test_sigma_scales(self):
        from common import sample_clipped_normal
        gen = torch.Generator().manual_seed(42)
        out = sample_clipped_normal(B=10000, sigma=0.1, generator=gen,
                                    clip_sigmas=3.0)
        # 经验 σ 应接近 0.1（截断略压缩 σ，差距 < 5%）
        assert abs(out.std().item() - 0.1) < 0.005

    def test_clip_at_three_sigma(self):
        from common import sample_clipped_normal
        gen = torch.Generator().manual_seed(42)
        out = sample_clipped_normal(B=100000, sigma=0.05, generator=gen,
                                    clip_sigmas=3.0)
        assert out.abs().max().item() <= 0.05 * 3.0 + 1e-7

    def test_zero_sigma_returns_zeros(self):
        from common import sample_clipped_normal
        out = sample_clipped_normal(B=64, sigma=0.0)
        assert torch.equal(out, torch.zeros(64))

    def test_generator_reproducible(self):
        from common import sample_clipped_normal
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = sample_clipped_normal(B=128, sigma=0.05, generator=g1)
        b = sample_clipped_normal(B=128, sigma=0.05, generator=g2)
        assert torch.equal(a, b)

    def test_no_grad(self):
        from common import sample_clipped_normal
        out = sample_clipped_normal(B=8, sigma=0.05)
        assert out.requires_grad is False
```

- [ ] **Step 2: 跑测试确认失败**

```
cd sim
python -m pytest tests/test_common.py::TestSampleClippedNormal -v
```

期望：`ImportError: cannot import name 'sample_clipped_normal'` 或全部失败。

- [ ] **Step 3: 实现 `sample_clipped_normal`**

把以下加到 `sim/common.py` 末尾：

```python
def sample_clipped_normal(B: int, sigma: float,
                          generator: torch.Generator | None = None,
                          clip_sigmas: float = 3.0) -> torch.Tensor:
    """单步独立高斯采样，clip_sigmas·σ 处截断；σ=0 时返回全零张量。

    Args:
        B: batch 维度，输出 shape = [B]。
        sigma: 高斯标准差。<= 0 时直接返回 torch.zeros(B)。
        generator: 可选 torch.Generator，传入则采样可复现。None 走全局随机。
        clip_sigmas: 截断倍数（默认 3σ）。

    Returns:
        shape [B] / dtype float32 / requires_grad=False 的张量。
    """
    if sigma <= 0:
        return torch.zeros(B, dtype=torch.float32)
    noise = torch.randn(B, generator=generator, dtype=torch.float32) * float(sigma)
    if clip_sigmas is not None and clip_sigmas > 0:
        bound = float(sigma) * float(clip_sigmas)
        noise = noise.clamp(-bound, bound)
    return noise
```

- [ ] **Step 4: 跑测试确认通过**

```
python -m pytest tests/test_common.py::TestSampleClippedNormal -v
```

期望：6 个测试全 PASS。

- [ ] **Step 5: 提交**

```bash
git add sim/common.py sim/tests/test_common.py
git commit -m "[sim] common 增加 sample_clipped_normal：单步截断高斯采样工具"
git push
```

---

## Task 2: yaml 配置段 `feedback_noise` / `command_dither`

**Files:**
- Modify: `sim/configs/default.yaml`（在 `domain_randomization` 段后追加）

- [ ] **Step 1: 在 `domain_randomization` 段之后追加两个新段**

打开 `sim/configs/default.yaml`，在第 111 行 `cfcr_range: 0.20` 之后插入：

```yaml

# 状态反馈噪声（仅 train_batch.py + truck_trailer 生效；hard_mode 验证默认关）
# 在控制器读取 vehicle 状态之前往真值上加独立高斯，模拟定位/IMU/轮速噪声。
# 单步白噪声、每 batch 元素独立、3σ 截断。loss 仍在 vehicle 真值上算。
feedback_noise:
  enable: false
  sigma_x_m: 0.02            # 位置 x 噪声 σ（合成出 lat_err σ ≈ 0.02 m）
  sigma_y_m: 0.02            # 位置 y 噪声 σ
  sigma_yaw_deg: 0.115       # 朝向噪声 σ（≈ 0.002 rad）
  sigma_speed_kph: 0.18      # 车速噪声 σ（≈ 0.05 m/s）
  sigma_yawrate_radps: 0.002 # 横摆率噪声 σ
  clip_sigmas: 3.0

# 指令高频抖动（同样 train_batch.py + truck_trailer，hard_mode 默认关）
# 控制器算出 delta/torque 后、送进 vehicle.step 之前加高斯抖动。
# history 仍记录控制器原始输出，loss 用真值。
command_dither:
  enable: false
  sigma_delta_rad: 0.001     # 前轮转角抖动 σ（≈ 0.057°）
  sigma_torque_nm: 15.0      # 总车轮扭矩抖动 σ
  clip_sigmas: 3.0
```

- [ ] **Step 2: 跑现有测试确认 yaml 加载未坏**

```
python -m pytest tests/test_train_batch.py -k "load or default" -v
```

如果有相关测试就跑；没有的话跑：

```
python -c "from config import load_config; cfg=load_config(); assert cfg['feedback_noise']['enable'] is False; assert cfg['command_dither']['enable'] is False; print('OK')"
```

期望：输出 `OK`。

- [ ] **Step 3: 提交**

```bash
git add sim/configs/default.yaml
git commit -m "[sim] default.yaml 新增 feedback_noise / command_dither 段（默认 enable=false）"
git push
```

---

## Task 3: 配置解析 helper `_resolve_noise_config` / `_resolve_dither_config`

**Files:**
- Modify: `sim/optim/train_batch.py`（在 `_resolve_dr_config` 附近追加）
- Create: `sim/tests/test_noise_dither.py`

- [ ] **Step 1: 写失败测试**

新建 `sim/tests/test_noise_dither.py`：

```python
"""噪声 + 指令抖动单元 / 集成测试。覆盖：

- sample_clipped_normal 已在 test_common.py 测过
- _resolve_noise_config / _resolve_dither_config 的 cfg+CLI 合并
- run_simulation_batch 在 noise/dither 关闭时与基线一致
- 同 seed 复现 / hard_mode 自动 mute
- train_batch 端到端 2 epoch 跑通
"""
from __future__ import annotations

import os
import sys
import math

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config, apply_plant_override
from model.trajectory import generate_circle
from optim.train_batch import (
    _resolve_noise_config, _resolve_dither_config,
    run_simulation_batch, train_batch)


def _truck_cfg():
    cfg = load_config()
    apply_plant_override(cfg, 'truck_trailer')
    cfg['truck_trailer_vehicle']['default_trailer_mass_kg'] = 0.0
    cfg['truck_trailer_vehicle']['checkpoint_path'] = ''
    return cfg


def _short_trajs():
    return [
        generate_circle(radius=15.0, speed=2.5, arc_angle=math.pi / 2),
        generate_circle(radius=25.0, speed=4.0, arc_angle=math.pi / 2),
    ]


class TestResolveNoiseConfig:
    def test_default_disabled(self):
        cfg = _truck_cfg()
        out = _resolve_noise_config(cfg, overrides=None)
        assert out['enable'] is False
        assert out['sigma_x_m'] == 0.02

    def test_cli_override_enables(self):
        cfg = _truck_cfg()
        out = _resolve_noise_config(cfg, overrides={'enable': True})
        assert out['enable'] is True

    def test_cli_sigma_override(self):
        cfg = _truck_cfg()
        out = _resolve_noise_config(cfg, overrides={'sigma_x_m': 0.10})
        assert out['sigma_x_m'] == 0.10
        assert out['sigma_y_m'] == 0.02  # 未覆盖项保留

    def test_none_overrides_keep_yaml(self):
        cfg = _truck_cfg()
        out = _resolve_noise_config(cfg, overrides={'enable': None,
                                                    'sigma_x_m': None})
        assert out['enable'] is False
        assert out['sigma_x_m'] == 0.02


class TestResolveDitherConfig:
    def test_default_disabled(self):
        cfg = _truck_cfg()
        out = _resolve_dither_config(cfg, overrides=None)
        assert out['enable'] is False
        assert out['sigma_delta_rad'] == 0.001
        assert out['sigma_torque_nm'] == 15.0

    def test_cli_overrides(self):
        cfg = _truck_cfg()
        out = _resolve_dither_config(cfg, overrides={
            'enable': True, 'sigma_torque_nm': 50.0})
        assert out['enable'] is True
        assert out['sigma_torque_nm'] == 50.0
        assert out['sigma_delta_rad'] == 0.001
```

- [ ] **Step 2: 跑测试确认失败**

```
python -m pytest tests/test_noise_dither.py::TestResolveNoiseConfig tests/test_noise_dither.py::TestResolveDitherConfig -v
```

期望：`ImportError` 或 6 个失败。

- [ ] **Step 3: 实现 `_resolve_noise_config` / `_resolve_dither_config`**

打开 `sim/optim/train_batch.py`，在 `_resolve_dr_config` 函数定义之后（约第 1320 行附近）追加：

```python
_NOISE_KEYS = ('enable', 'sigma_x_m', 'sigma_y_m', 'sigma_yaw_deg',
               'sigma_speed_kph', 'sigma_yawrate_radps', 'clip_sigmas')

_DITHER_KEYS = ('enable', 'sigma_delta_rad', 'sigma_torque_nm', 'clip_sigmas')


def _resolve_noise_config(cfg: dict, overrides: dict | None = None) -> dict:
    """合并 cfg['feedback_noise'] 与 CLI overrides。CLI 非 None 时优先。"""
    base = dict(cfg.get('feedback_noise', {}))
    out = {k: base.get(k) for k in _NOISE_KEYS}
    if overrides:
        for k in _NOISE_KEYS:
            v = overrides.get(k)
            if v is not None:
                out[k] = v
    # 默认值兜底（万一 yaml 缺字段）
    out.setdefault('enable', False)
    out.setdefault('sigma_x_m', 0.02)
    out.setdefault('sigma_y_m', 0.02)
    out.setdefault('sigma_yaw_deg', 0.115)
    out.setdefault('sigma_speed_kph', 0.18)
    out.setdefault('sigma_yawrate_radps', 0.002)
    out.setdefault('clip_sigmas', 3.0)
    return out


def _resolve_dither_config(cfg: dict, overrides: dict | None = None) -> dict:
    """合并 cfg['command_dither'] 与 CLI overrides。CLI 非 None 时优先。"""
    base = dict(cfg.get('command_dither', {}))
    out = {k: base.get(k) for k in _DITHER_KEYS}
    if overrides:
        for k in _DITHER_KEYS:
            v = overrides.get(k)
            if v is not None:
                out[k] = v
    out.setdefault('enable', False)
    out.setdefault('sigma_delta_rad', 0.001)
    out.setdefault('sigma_torque_nm', 15.0)
    out.setdefault('clip_sigmas', 3.0)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

```
python -m pytest tests/test_noise_dither.py::TestResolveNoiseConfig tests/test_noise_dither.py::TestResolveDitherConfig -v
```

期望：6 个测试全 PASS。

- [ ] **Step 5: 提交**

```bash
git add sim/optim/train_batch.py sim/tests/test_noise_dither.py
git commit -m "[sim] train_batch 增加 _resolve_noise_config / _resolve_dither_config 合并 cfg+CLI"
git push
```

---

## Task 4: `run_simulation_batch` 注入状态噪声 + 指令抖动

**Files:**
- Modify: `sim/optim/train_batch.py`（`run_simulation_batch` + `_run_sim_batch_inner`）
- Modify: `sim/tests/test_noise_dither.py`（追加 `TestRunSimulationBatchNoise`）

- [ ] **Step 1: 写失败测试**

把以下追加到 `sim/tests/test_noise_dither.py`：

```python
class TestRunSimulationBatchNoise:
    def test_disabled_path_unchanged(self):
        """noise/dither 关闭 + 同 seed 路径下，两次仿真输出逐元素相等。"""
        cfg = _truck_cfg()
        trajs = _short_trajs()
        # 跑两次（不传 noise_params / dither_params），结果应该一致（无随机源）
        out1 = run_simulation_batch(trajs, cfg=cfg, hard_mode=True,
                                    noise_params=None, dither_params=None)
        out2 = run_simulation_batch(trajs, cfg=cfg, hard_mode=True,
                                    noise_params=None, dither_params=None)
        assert torch.equal(out1['x'], out2['x'])
        assert torch.equal(out1['y'], out2['y'])
        assert torch.equal(out1['steer'], out2['steer'])

    def test_noise_changes_output(self):
        """noise enable + 不同 seed → 输出应有差异。"""
        cfg = _truck_cfg()
        trajs = _short_trajs()
        baseline = run_simulation_batch(trajs, cfg=cfg, hard_mode=True,
                                        noise_params=None)
        np = {'enable': True, 'sigma_x_m': 0.05, 'sigma_y_m': 0.05,
              'sigma_yaw_deg': 0.5, 'sigma_speed_kph': 0.5,
              'sigma_yawrate_radps': 0.01, 'clip_sigmas': 3.0,
              'generator': torch.Generator().manual_seed(11)}
        noisy = run_simulation_batch(trajs, cfg=cfg, hard_mode=False,
                                     noise_params=np)
        assert not torch.allclose(baseline['x'], noisy['x'], atol=1e-6)

    def test_noise_seeded_reproducible(self):
        """同 seed 噪声两次仿真完全一致。"""
        cfg = _truck_cfg()
        trajs = _short_trajs()
        def _np(seed):
            return {'enable': True, 'sigma_x_m': 0.05, 'sigma_y_m': 0.05,
                    'sigma_yaw_deg': 0.5, 'sigma_speed_kph': 0.5,
                    'sigma_yawrate_radps': 0.01, 'clip_sigmas': 3.0,
                    'generator': torch.Generator().manual_seed(seed)}
        a = run_simulation_batch(trajs, cfg=cfg, hard_mode=False,
                                 noise_params=_np(42))
        b = run_simulation_batch(trajs, cfg=cfg, hard_mode=False,
                                 noise_params=_np(42))
        assert torch.equal(a['x'], b['x'])
        assert torch.equal(a['steer'], b['steer'])

    def test_dither_changes_output_but_history_unchanged(self):
        """dither 影响 plant 状态（vehicle.x 改变），但 history 的 steer/torque
        是控制器原始输出——同 cfg+seed 下，控制器对同样输入的输出应一致；这里
        因为 plant 反馈不同所以下一步 steer 也变了，故只断言 'plant 状态变了'
        即可。"""
        cfg = _truck_cfg()
        trajs = _short_trajs()
        baseline = run_simulation_batch(trajs, cfg=cfg, hard_mode=False,
                                        noise_params=None, dither_params=None)
        dp = {'enable': True, 'sigma_delta_rad': 0.005,
              'sigma_torque_nm': 50.0, 'clip_sigmas': 3.0,
              'generator': torch.Generator().manual_seed(7)}
        out = run_simulation_batch(trajs, cfg=cfg, hard_mode=False,
                                   noise_params=None, dither_params=dp)
        assert not torch.allclose(baseline['x'], out['x'], atol=1e-6)

    def test_hard_mode_mutes_noise(self):
        """hard_mode=True 时即使传入 noise_params 也强制 mute，输出与无噪一致。"""
        cfg = _truck_cfg()
        trajs = _short_trajs()
        clean = run_simulation_batch(trajs, cfg=cfg, hard_mode=True,
                                     noise_params=None)
        np_aggressive = {'enable': True, 'sigma_x_m': 0.5, 'sigma_y_m': 0.5,
                         'sigma_yaw_deg': 5.0, 'sigma_speed_kph': 5.0,
                         'sigma_yawrate_radps': 0.5, 'clip_sigmas': 3.0,
                         'generator': torch.Generator().manual_seed(99)}
        muted = run_simulation_batch(trajs, cfg=cfg, hard_mode=True,
                                     noise_params=np_aggressive)
        assert torch.equal(clean['x'], muted['x'])
        assert torch.equal(clean['steer'], muted['steer'])
```

- [ ] **Step 2: 跑测试确认失败**

```
python -m pytest tests/test_noise_dither.py::TestRunSimulationBatchNoise -v
```

期望：`TypeError: ... unexpected keyword 'noise_params'` 或全部失败。

- [ ] **Step 3: 修改 `run_simulation_batch` 函数签名**

打开 `sim/optim/train_batch.py`，找到 `def run_simulation_batch(...)`（约第 1024 行）。修改签名 + docstring：

```python
def run_simulation_batch(trajectories: list, cfg: dict = None,
                         lat_ctrl: BatchedLatTruck = None,
                         lon_ctrl: BatchedLonCtrl = None,
                         tbptt_k: int = 0,
                         hard_mode: bool = False,
                         trailer_mass_kg=None,
                         domain_params: dict | None = None,
                         noise_params: dict | None = None,
                         dither_params: dict | None = None) -> dict:
    """B 条轨迹同步推进 50Hz 闭环。支持 truck_trailer / hybrid_dynamic plant。

    Args:
        hard_mode: True 时控制器走硬限幅路径，自动 torch.no_grad()。**hard_mode
            =True 时即使传入 noise_params/dither_params 也强制 mute**——验证
            路径只看真值。
        noise_params: 可选 dict，包含 enable / sigma_{x_m,y_m,yaw_deg,
            speed_kph,yawrate_radps} / clip_sigmas / generator。enable=False
            或 None 时不注入。开启时在控制器读取 vehicle 状态之前往真值上加
            独立高斯。
        dither_params: 可选 dict，包含 enable / sigma_{delta_rad,torque_nm} /
            clip_sigmas / generator。开启时在 vehicle.step 收到指令之前加
            高斯抖动。history 记录的是控制器原始输出（不含 dither）。

    （其它字段不变）
    """
```

然后修改函数体——把 `noise_params` / `dither_params` 透传给 `_run_sim_batch_inner`：

```python
    grad_ctx = torch.no_grad() if hard_mode else _nullctx()
    with grad_ctx:
        return _run_sim_batch_inner(
            trajectories, cfg, lat_ctrl, lon_ctrl, tbptt_k,
            hard_mode, trailer_mass_kg, domain_params,
            noise_params, dither_params)
```

- [ ] **Step 4: 修改 `_run_sim_batch_inner` 注入噪声**

找到 `def _run_sim_batch_inner(...)`（约第 1070 行）。修改签名加两个参数：

```python
def _run_sim_batch_inner(trajectories, cfg, lat_ctrl, lon_ctrl, tbptt_k,
                         hard_mode, trailer_mass_override,
                         domain_params=None,
                         noise_params=None, dither_params=None):
```

在函数顶部、`prev_steer = torch.zeros(B)` 之前（约第 1126 行附近）加噪声 active 判定：

```python
    # hard_mode 强制 mute——验证路径只看真值
    noise_active = (not hard_mode and noise_params is not None
                    and noise_params.get('enable', False))
    dither_active = (not hard_mode and dither_params is not None
                     and dither_params.get('enable', False))
    if noise_active:
        n_gen = noise_params.get('generator')
        n_sx = float(noise_params['sigma_x_m'])
        n_sy = float(noise_params['sigma_y_m'])
        n_syaw = float(noise_params['sigma_yaw_deg'])
        n_sv = float(noise_params['sigma_speed_kph'])
        n_syr = float(noise_params['sigma_yawrate_radps'])
        n_clip = float(noise_params.get('clip_sigmas', 3.0))
    if dither_active:
        d_gen = dither_params.get('generator')
        d_sd = float(dither_params['sigma_delta_rad'])
        d_st = float(dither_params['sigma_torque_nm'])
        d_clip = float(dither_params.get('clip_sigmas', 3.0))
```

需要在文件顶部 `from common import ...` 那行（如果未导入 `sample_clipped_normal`）追加：

```python
from common import sample_clipped_normal
```

接着，在主循环 `for step in range(T_max):` 内部，找到 `yawrate = vehicle.yawrate` 那一行（约第 1141 行）。**替换**该行起到 `lon_ctrl.compute(...)` 调用结束的整段，改为：

```python
        # 控制器读取的状态：默认真值，noise_active 时加独立高斯
        yawrate = vehicle.yawrate
        if noise_active:
            x_meas = vehicle.x + sample_clipped_normal(B, n_sx, n_gen, n_clip)
            y_meas = vehicle.y + sample_clipped_normal(B, n_sy, n_gen, n_clip)
            yaw_deg_meas = vehicle.yaw_deg + sample_clipped_normal(
                B, n_syaw, n_gen, n_clip)
            speed_kph_meas = vehicle.speed_kph + sample_clipped_normal(
                B, n_sv, n_gen, n_clip)
            yawrate_meas = yawrate + sample_clipped_normal(
                B, n_syr, n_gen, n_clip)
        else:
            x_meas, y_meas = vehicle.x, vehicle.y
            yaw_deg_meas, speed_kph_meas = vehicle.yaw_deg, vehicle.speed_kph
            yawrate_meas = yawrate

        steer_out, _kappa_cur, _nk, curvature_far, steer_fb, steer_ff = \
            lat_ctrl.compute(
                x=x_meas, y=y_meas,
                yaw_deg=yaw_deg_meas, speed_kph=speed_kph_meas,
                yawrate=yawrate_meas, steer_feedback=prev_steer,
                btraj=bt, dt=dt, hard_mode=hard_mode)

        acc_cmd = lon_ctrl.compute(
            x=x_meas, y=y_meas,
            yaw_deg=yaw_deg_meas, speed_kph=speed_kph_meas,
            curvature_far=curvature_far,
            btraj=bt, t_now=t_now,
            ctrl_first_active=(step == 0), dt=dt, hard_mode=hard_mode)
```

然后找到 `delta_front = steer_out / steer_ratio * DEG2RAD` 后的一段（torque_wheel 计算），保持不变。

接下来，找到 `vehicle.step(delta=delta_front, torque_wheel=torque_wheel)`（约第 1185 行）。把 `vehicle.step(...)` 这一行改为：

```python
        # 指令送 plant 之前加 dither（history 仍记控制器原始输出）
        if dither_active:
            delta_to_plant = delta_front + sample_clipped_normal(
                B, d_sd, d_gen, d_clip)
            torque_to_plant = torque_wheel + sample_clipped_normal(
                B, d_st, d_gen, d_clip)
        else:
            delta_to_plant, torque_to_plant = delta_front, torque_wheel

        v_prev = vehicle.v.detach()
        vehicle.step(delta=delta_to_plant, torque_wheel=torque_to_plant)
        prev_steer = steer_out
```

（注：原 `v_prev = vehicle.v.detach()` 必须保留在 `vehicle.step` 之前，已包含在上面）。

- [ ] **Step 5: 跑测试确认通过**

```
python -m pytest tests/test_noise_dither.py::TestRunSimulationBatchNoise -v
```

期望：5 个测试全 PASS。

- [ ] **Step 6: 跑现有 truck_trailer / DR 测试确保未破坏**

```
python -m pytest tests/test_train_batch.py tests/test_domain_randomization.py tests/test_truck_trailer_vehicle.py -q
```

期望：全部 PASS（噪声/抖动默认 None，路径行为不变）。

- [ ] **Step 7: 提交**

```bash
git add sim/optim/train_batch.py sim/tests/test_noise_dither.py
git commit -m "[sim] run_simulation_batch 注入状态噪声 + 指令抖动；hard_mode 强制 mute"
git push
```

---

## Task 5: `train_batch()` 函数串联噪声/抖动配置 + 锁种子

**Files:**
- Modify: `sim/optim/train_batch.py`（`train_batch` 函数签名 + 内部 epoch 循环）
- Modify: `sim/tests/test_noise_dither.py`（追加 `TestTrainBatchNoiseSeed`）

- [ ] **Step 1: 写失败测试**

追加到 `sim/tests/test_noise_dither.py`：

```python
class TestTrainBatchNoiseSeed:
    @pytest.mark.slow
    def test_train_batch_noise_disabled_unchanged(self):
        """传 noise_overrides=None / dither_overrides=None 时，train_batch 与原路径 2 epoch 结果一致。"""
        out_a = train_batch(trajectories=['lane_change'], n_epochs=2,
                            plant='truck_trailer', config_path=None,
                            disable_mlp=True,
                            dr_overrides={'enable': False},
                            noise_overrides=None,
                            dither_overrides=None)
        out_b = train_batch(trajectories=['lane_change'], n_epochs=2,
                            plant='truck_trailer', config_path=None,
                            disable_mlp=True,
                            dr_overrides={'enable': False})
        assert abs(out_a['losses'][-1] - out_b['losses'][-1]) < 1e-6

    @pytest.mark.slow
    def test_train_batch_seeded_reproducible(self):
        """相同 noise_seed 跑 2 epoch，最终 loss 完全一致。"""
        kw = dict(trajectories=['lane_change'], n_epochs=2,
                  plant='truck_trailer', config_path=None,
                  disable_mlp=True,
                  dr_overrides={'enable': True, 'K': 2},
                  noise_overrides={'enable': True},
                  dither_overrides={'enable': True},
                  dr_seed=2026, noise_seed=2026)
        a = train_batch(**kw)
        b = train_batch(**kw)
        assert abs(a['losses'][-1] - b['losses'][-1]) < 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

```
python -m pytest tests/test_noise_dither.py::TestTrainBatchNoiseSeed -v -m slow
```

期望：`TypeError: train_batch() got an unexpected keyword argument 'noise_overrides'` 或失败。

- [ ] **Step 3: 修改 `train_batch()` 函数签名**

找到 `def train_batch(...)`（约第 1345 行）。在 `dr_seed: int | None = None` 之后插入两行：

```python
def train_batch(trajectories=None,
                # ... 已有参数原样保留 ...
                dr_overrides: dict | None = None,
                disable_mlp: bool = False,
                dr_seed: int | None = None,
                noise_overrides: dict | None = None,
                dither_overrides: dict | None = None,
                noise_seed: int | None = None):
```

（其它已有参数不动；只在末尾追加 noise/dither/seed 三个）

- [ ] **Step 4: 在 `train_batch` 函数内部解析 noise/dither config + 建 Generator**

找到 `dr_config = _resolve_dr_config(cfg, dr_overrides)`（约第 1368 行）。在它之后追加：

```python
    noise_config = _resolve_noise_config(cfg, noise_overrides)
    dither_config = _resolve_dither_config(cfg, dither_overrides)
```

找到 `dr_generator = (torch.Generator()...) if dr_seed is not None else None)`（约第 1426 行）。在它之后追加：

```python
        noise_generator = (torch.Generator().manual_seed(int(noise_seed))
                           if noise_seed is not None else None)
```

注意 noise/dither **共用同一个 generator**——这样既保证 noise_seed 是单一锁种子入口，也保证 7 个采样调用顺序确定。我们在 noise_params 和 dither_params 里都填同一个 `noise_generator`：

```python
        noise_params_for_run = None
        dither_params_for_run = None
        if noise_config['enable']:
            noise_params_for_run = dict(noise_config)
            noise_params_for_run['generator'] = noise_generator
        if dither_config['enable']:
            dither_params_for_run = dict(dither_config)
            dither_params_for_run['generator'] = noise_generator
```

把这段紧跟在 `noise_generator = ...` 之后。

- [ ] **Step 5: 把 noise/dither_params 透传到 `run_simulation_batch` 调用**

找到 epoch 循环里调用 `run_simulation_batch(...)` 的那行（约第 1479 行）。修改为：

```python
        history = run_simulation_batch(
            train_trajs, cfg=cfg, lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
            tbptt_k=tbptt_k, domain_params=domain_params,
            noise_params=noise_params_for_run,
            dither_params=dither_params_for_run)
```

如果在同一函数内还有别处调用 `run_simulation_batch`（比如 epoch 0 baseline 评估），同样透传。**但 hard_mode=True 的调用不用传 noise/dither**——hard_mode 内部会强制 mute，传也无效。

- [ ] **Step 6: 把 noise/dither config 写进训练日志/result**

找到 `train_batch` 函数返回 result dict 的位置（约第 1691 行 `dr_seed`：）。在 dr_seed 字段附近追加：

```python
        'noise_config': dict(noise_config),
        'dither_config': dict(dither_config),
        'noise_seed': noise_seed,
```

- [ ] **Step 7: 跑测试确认通过**

```
python -m pytest tests/test_noise_dither.py::TestTrainBatchNoiseSeed -v -m slow
```

期望：2 个测试 PASS（每个 ~3-5 min on CPU）。

- [ ] **Step 8: 跑全套 batch 训练相关测试**

```
python -m pytest tests/test_train_batch.py tests/test_domain_randomization.py tests/test_noise_dither.py -q
```

期望：全部 PASS（带 slow 的可能跳过，那些非 slow 必须 PASS）。

- [ ] **Step 9: 提交**

```bash
git add sim/optim/train_batch.py sim/tests/test_noise_dither.py
git commit -m "[sim] train_batch 串联 noise/dither config 与 noise_seed，复用单 Generator 锁顺序"
git push
```

---

## Task 6: CLI flags

**Files:**
- Modify: `sim/optim/train_batch.py`（`__main__` argparse 段）

- [ ] **Step 1: 在 argparse 段追加 noise/dither flags**

找到 `parser.add_argument('--dr-seed', ...)`（约第 1758 行）。在它**之后**、`--disable-mlp` **之前**追加：

```python
    # 状态反馈噪声（CLI 仅作覆盖；默认走 cfg['feedback_noise']）
    parser.add_argument('--noise-enable', action='store_true', default=None,
                        help='启用状态反馈噪声')
    parser.add_argument('--no-noise', dest='noise_enable',
                        action='store_false', default=None,
                        help='强制关闭状态反馈噪声（覆盖 yaml）')
    parser.add_argument('--sigma-x', type=float, default=None,
                        help='位置 x 噪声 σ (m)')
    parser.add_argument('--sigma-y', type=float, default=None,
                        help='位置 y 噪声 σ (m)')
    parser.add_argument('--sigma-yaw', type=float, default=None,
                        help='朝向噪声 σ (deg)')
    parser.add_argument('--sigma-speed', type=float, default=None,
                        help='车速噪声 σ (km/h)')
    parser.add_argument('--sigma-yawrate', type=float, default=None,
                        help='横摆率噪声 σ (rad/s)')
    # 指令抖动
    parser.add_argument('--dither-enable', action='store_true', default=None,
                        help='启用指令高频抖动')
    parser.add_argument('--no-dither', dest='dither_enable',
                        action='store_false', default=None,
                        help='强制关闭指令抖动')
    parser.add_argument('--sigma-delta', type=float, default=None,
                        help='delta 抖动 σ (rad)')
    parser.add_argument('--sigma-torque', type=float, default=None,
                        help='torque 抖动 σ (N·m)')
    parser.add_argument('--noise-seed', type=int, default=None,
                        help='噪声 + 抖动共用的随机种子（None 不固定）')
```

- [ ] **Step 2: 在 `dr_overrides = {...}` 之后构建 noise/dither overrides**

找到 `dr_overrides = {...}`（约第 1764 行）。在其下追加：

```python
    noise_overrides = {
        'enable': args.noise_enable,
        'sigma_x_m': args.sigma_x,
        'sigma_y_m': args.sigma_y,
        'sigma_yaw_deg': args.sigma_yaw,
        'sigma_speed_kph': args.sigma_speed,
        'sigma_yawrate_radps': args.sigma_yawrate,
    }
    dither_overrides = {
        'enable': args.dither_enable,
        'sigma_delta_rad': args.sigma_delta,
        'sigma_torque_nm': args.sigma_torque,
    }
```

- [ ] **Step 3: 把 overrides + noise_seed 透传到 `train_batch(...)` 调用**

找到 `result = train_batch(...)`（约第 1771 行）。在调用最末尾追加：

```python
    result = train_batch(
        # ... 已有参数原样保留 ...
        dr_overrides=dr_overrides,
        disable_mlp=args.disable_mlp,
        dr_seed=args.dr_seed,
        noise_overrides=noise_overrides,
        dither_overrides=dither_overrides,
        noise_seed=args.noise_seed)
```

- [ ] **Step 4: 把 noise/dither config 写进 post_training hyperparams**

找到 `hyperparams = {...}` 字典（约第 1800 行）。在 `'dr_seed': args.dr_seed,` 之后追加：

```python
            'feedback_noise': result.get('noise_config'),
            'command_dither': result.get('dither_config'),
            'noise_seed': args.noise_seed,
```

- [ ] **Step 5: 烟测 CLI 解析**

```
python -m sim.optim.train_batch --help 2>&1 | grep -E "noise|dither|sigma" | head -20
```

期望：能看到 `--noise-enable`、`--dither-enable`、`--sigma-x`、`--noise-seed` 等条目。

```
python optim/train_batch.py --noise-enable --dither-enable --noise-seed 42 --dr-enable --dr-seed 42 --epochs 1 --trajectories lane_change --disable-mlp 2>&1 | head -40
```

注：实际执行需要在 sim/ 目录下；上面只是为了示意 CLI 形态。如果想烟测但不想等完整 1 epoch，可以做：

```
python -c "
import sys; sys.argv = ['train_batch', '--noise-enable', '--sigma-x', '0.05', '--noise-seed', '42', '--epochs', '1', '--trajectories', 'lane_change', '--disable-mlp', '--no-post-training']
import argparse
" 
```

只验 argparse 解析无错即可。

- [ ] **Step 6: 提交**

```bash
git add sim/optim/train_batch.py
git commit -m "[sim] train_batch CLI 增加 --noise-enable / --dither-enable / --sigma-* / --noise-seed"
git push
```

---

## Task 7: 文档同步

**Files:**
- Modify: `sim/CLAUDE.md`
- Modify: `README.md`（项目根，如有 DR 段需扩列）

- [ ] **Step 1: sim/CLAUDE.md 域随机化段补噪声/抖动说明**

打开 `sim/CLAUDE.md`，找到 `## 域随机化 (--dr-enable，仅 train_batch.py + truck_trailer)` 段（约第 239 行）。在其末尾追加一段：

```markdown
### 状态反馈噪声 + 指令抖动（叠加在 DR 之上）

`--noise-enable` 在控制器读取 vehicle 状态之前往真值上加独立高斯（5 通道：x/y/yaw/speed/yawrate），`--dither-enable` 在 vehicle.step 收到指令之前给 delta/torque 加高斯抖动；两者均为单步白噪声、3σ 截断。loss 始终用 vehicle 真值，hard_mode 验证路径强制 mute——所以 V1 49 场景对比仍然干净。

yaml 配置入口 `feedback_noise` / `command_dither`，CLI 优先级高于 yaml；`--noise-seed` 锁噪声采样种子（与 `--dr-seed` 解耦）。详见 `docs/plans/2026-05-08-aggressive-dr-noise-dither-design.md`。
```

- [ ] **Step 2: sim/CLAUDE.md 常用命令段补一行**

找到常用命令表（约第 88 行 `--dr-enable` 那行附近）。追加一行：

```bash
python optim/train_batch.py --plant truck_trailer --dr-enable --noise-enable --dither-enable --dr-seed 2026 --noise-seed 2026 --epochs 6  # 激进 DR：物理 + 噪声 + 抖动叠加训练
```

- [ ] **Step 3: README.md 文档索引补一条**

打开项目根 `README.md`。找到 DR 设计文档所在的索引段（commit `1084b79` 加的那条）。在其下追加：

```markdown
- [激进 DR：噪声 + 抖动设计](docs/plans/2026-05-08-aggressive-dr-noise-dither-design.md) — 在保守档 DR 上叠加状态噪声 + 指令抖动
- [激进 DR：噪声 + 抖动实施计划](docs/plans/2026-05-08-aggressive-dr-noise-dither-plan.md)
```

如果 README 现在没有 DR 段，找一个合适的位置插在"训练 / 调参"段附近即可。

- [ ] **Step 4: 提交**

```bash
git add sim/CLAUDE.md README.md
git commit -m "[docs] sim/CLAUDE.md + README 同步噪声/抖动 CLI 与设计文档索引"
git push
```

---

## Task 8: 端到端：2 epoch 烟测 + 6 epoch 完整训练

**Files:**
- Modify: `docs/plans/2026-05-08-aggressive-dr-noise-dither-design.md`（在末尾追加"训练结果"段）

- [ ] **Step 1: 2 epoch 烟测**

```
cd sim
python -u optim/train_batch.py --plant truck_trailer \
    --dr-enable --dr-seed 2026 \
    --noise-enable --dither-enable --noise-seed 2026 \
    --disable-mlp --epochs 2 --no-post-training
```

检查标准（不通过则定位修复）：
- 没有 NaN 梯度
- 每 epoch 都打印出 loss、lat_rmse、grad_norm
- 训练完成、tuned yaml 保存到 `sim/configs/tuned/tuned_*.yaml`

- [ ] **Step 2: 6 epoch 完整训练 + post_training**

```
python -u optim/train_batch.py --plant truck_trailer \
    --dr-enable --dr-seed 2026 \
    --noise-enable --dither-enable --noise-seed 2026 \
    --disable-mlp --epochs 6 2>&1 | tee sim/results/training/aggressive_dr_run.log
```

期望耗时 ~50 min（DR + noise 比纯 DR 略慢 5-10%）。完成后：
- `sim/configs/tuned/tuned_*.yaml` 含本次 tuned 参数
- `sim/results/training/truck_trailer/<timestamp>/` 含 49 场景对比图、loss 曲线、experiment_log.yaml
- 49 场景对比 V1 lat_rmse 改善 / 退化对照（与既有 DR-only 数据对照）

- [ ] **Step 3: 把训练结果写回设计文档**

打开 `docs/plans/2026-05-08-aggressive-dr-noise-dither-design.md`，在末尾追加 "## 训练结果" 段，照 2026-05-08-domain-randomization-design.md 的"训练结果"段格式（commit hash、loss 表、49 场景对照、关键参数变化、产物路径）填实际数字。

- [ ] **Step 4: 提交训练记录 + 设计文档更新**

`results/training/<timestamp>/` 是 .gitignore 排除的，不入仓。把日志摘要 + 关键图截到设计文档里：

```bash
git add docs/plans/2026-05-08-aggressive-dr-noise-dither-design.md sim/configs/tuned/tuned_*.yaml
git commit -m "[sim] 激进 DR 首跑：物理 + 噪声 + 抖动叠加 6 epoch 结果"
git push
```

---

## Self-Review

实施前自检（写完后用过一遍）：

**Spec coverage**：
- 4 通道状态噪声 (lat/head/v/yawrate) ✓ Task 4 注入 5 物理通道（x/y 合成 lat、yaw=head、speed=v、yawrate）
- 2 通道指令抖动 (delta/torque) ✓ Task 4
- 单步白噪声 + 3σ 截断 + 均值 0 ✓ Task 1 sample_clipped_normal
- 与 DR 共存 ✓ Task 5（dr_overrides 与 noise/dither_overrides 并行）
- yaml + CLI ✓ Task 2/6
- noise_seed 复现 ✓ Task 5（`test_train_batch_seeded_reproducible`）
- hard_mode 默认关噪 ✓ Task 4（`test_hard_mode_mutes_noise`）
- loss 用真值 ✓ Task 4（lateral_error/heading_error 计算位置不动）
- history 记控制器原始输出 ✓ Task 4（h_steer / h_torque 用 steer_out / torque_wheel）
- MLP 与噪声开关正交 ✓ Task 5（不动 disable_mlp 路径）
- 仅 train_batch + truck_trailer ✓ Task 4（不改 sim_loop scalar，不动 hybrid_*）

**Placeholder scan**：
- 所有 Step 都给出确切代码 ✓
- 测试代码完整可执行 ✓
- 命令行带绝对/相对路径 ✓
- 无 "TBD"、"add appropriate handling"、"similar to Task N"

**Type consistency**：
- `sample_clipped_normal(B, sigma, generator, clip_sigmas)` 在 Task 1/4/5 一致
- `noise_params` dict 字段名 (`sigma_x_m`, `sigma_y_m`, ...) 在 yaml/Task 3/4/5 一致
- `dither_params` dict 字段 (`sigma_delta_rad`, `sigma_torque_nm`) 一致
- `_resolve_noise_config` / `_resolve_dither_config` 签名 / 返回 dict 字段一致

**Scope check**：8 个 task 一条链路下来，每个 task 内部 5-9 步，总改动量 ~150 行代码 + ~250 行测试 + ~50 行 yaml/docs。可独立提交、单测保护、回归测试覆盖关闭路径。一个 plan 范围合适。
