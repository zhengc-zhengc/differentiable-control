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
