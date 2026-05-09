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


class TestRunSimulationBatchNoise:
    def test_disabled_path_unchanged(self):
        """noise/dither 关闭 + 同 cfg 路径下，两次仿真输出逐元素相等。"""
        cfg = _truck_cfg()
        trajs = _short_trajs()
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
        baseline = run_simulation_batch(trajs, cfg=cfg, hard_mode=False,
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

    def test_dither_changes_plant_state(self):
        """dither 注入会让 plant 状态偏离基线（vehicle.x 不一样）。
        history 的 steer/torque 是控制器原始输出——但因为 plant 反馈不同，
        下一步控制器输入也变，故 steer 序列也会变。这里只断言 plant 状态偏离。"""
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


class TestTrainBatchNoiseSeed:
    def test_train_batch_noise_disabled_unchanged(self, tmp_path, monkeypatch):
        """传 noise_overrides=None / dither_overrides=None 时 train_batch 与原路径 2 epoch 结果一致。"""
        monkeypatch.chdir(tmp_path)
        torch.manual_seed(7)
        out_a = train_batch(trajectories=_short_trajs(), n_epochs=2,
                            plant='truck_trailer', verbose=False,
                            disable_mlp=True,
                            dr_overrides={'enable': False},
                            noise_overrides=None,
                            dither_overrides=None,
                            param_snapshot_interval=0)
        torch.manual_seed(7)
        out_b = train_batch(trajectories=_short_trajs(), n_epochs=2,
                            plant='truck_trailer', verbose=False,
                            disable_mlp=True,
                            dr_overrides={'enable': False},
                            param_snapshot_interval=0)
        assert abs(out_a['losses'][-1] - out_b['losses'][-1]) < 1e-6

    def test_train_batch_seeded_reproducible(self, tmp_path, monkeypatch):
        """相同 noise_seed 跑 2 epoch，最终 loss 完全一致。"""
        monkeypatch.chdir(tmp_path)
        kw = dict(trajectories=_short_trajs(), n_epochs=2,
                  plant='truck_trailer', verbose=False,
                  disable_mlp=True,
                  dr_overrides={'enable': True, 'K': 2,
                                'mt_range': 0.05, 'cfcr_range': 0.10},
                  noise_overrides={'enable': True},
                  dither_overrides={'enable': True},
                  dr_seed=2026, noise_seed=2026,
                  param_snapshot_interval=0)
        torch.manual_seed(7)
        a = train_batch(**kw)
        torch.manual_seed(7)
        b = train_batch(**kw)
        assert abs(a['losses'][-1] - b['losses'][-1]) < 1e-6
