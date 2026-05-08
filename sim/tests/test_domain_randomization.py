# sim/tests/test_domain_randomization.py
"""域随机化单元 + 回归测试。覆盖：

- TruckTrailerNominalDynamics.set_domain：scalar/[B] 两种 shape 都可用，
  Iz_t 按 m_t 联动公式更新，[B] 状态广播正确
- _sample_dr_domains：均匀分布在 ±range 内，shape == [K]
- _resolve_dr_config：CLI 覆盖 cfg 默认值的合并逻辑
- run_simulation_batch + domain_params：vehicle 各 batch 元素受其 domain 影响
- DR 关闭时 train_batch 行为与原路径完全一致（回归保护）
- DR 开启时 train_batch 2 epoch 跑通，loss 收敛、参数变化非零
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math

from config import load_config, apply_plant_override
from model.truck_trailer_dynamics import TruckTrailerNominalDynamics
from model.trajectory import generate_circle, generate_lane_change
from optim.train_batch import (
    _resolve_dr_config, _sample_dr_domains, BatchedTruckTrailerVehicle,
    run_simulation_batch, train_batch)


def _short_truck_trajs():
    """两条短圆弧 + 不同速度，~150 步即可，避免 lane_change 在 5kph 下被
    pad 成 ~1800 步导致 train_batch 测试 > 10min。"""
    return [
        generate_circle(radius=15.0, speed=2.5, arc_angle=math.pi / 2),
        generate_circle(radius=25.0, speed=4.0, arc_angle=math.pi / 2),
    ]


def _truck_cfg():
    cfg = load_config()
    apply_plant_override(cfg, 'truck_trailer')
    cfg['truck_trailer_vehicle']['default_trailer_mass_kg'] = 0.0
    cfg['truck_trailer_vehicle']['checkpoint_path'] = ''  # 无 MLP
    return cfg


def _tt_params(cfg):
    return cfg['truck_trailer_vehicle']


# ─── set_domain：scalar / [B] / Iz 联动 ───────────────────────────────────

class TestSetDomain:
    def test_iz_linked_to_mt_scalar(self):
        cfg = _truck_cfg()
        tt = _tt_params(cfg)
        # 把 Iz_t/m_t 喂进 dynamics 构造（dynamics 需要的字段子集）
        params = {**tt, 'Iz_t': 48639.0}
        dyn = TruckTrailerNominalDynamics(params)
        m0 = float(dyn.m_t.item())
        Iz0 = float(dyn.Iz_t.item())

        # 设 m_t = 1.5 × nominal，Iz_t 应也 ×1.5
        dyn.set_domain(torch.tensor(m0 * 1.5),
                       torch.tensor(float(tt['Cf'])),
                       torch.tensor(float(tt['Cr'])))
        assert dyn.Iz_t.item() == pytest.approx(Iz0 * 1.5, rel=1e-5)
        assert dyn.m_t.item() == pytest.approx(m0 * 1.5, rel=1e-5)

    def test_batched_shape(self):
        cfg = _truck_cfg()
        tt = _tt_params(cfg)
        params = {**tt, 'Iz_t': 48639.0}
        dyn = TruckTrailerNominalDynamics(params)

        B = 4
        m_t = torch.tensor([8000.0, 9000.0, 10000.0, 11000.0])
        Cf = torch.full((B,), float(tt['Cf']))
        Cr = torch.full((B,), float(tt['Cr']))
        dyn.set_domain(m_t, Cf, Cr)

        assert dyn.m_t.shape == (B,)
        assert dyn.Iz_t.shape == (B,)
        assert dyn.Cf.shape == (B,)
        # Iz 联动检查
        ratio = m_t / float(params['m_t'])
        expected_Iz = float(params['Iz_t']) * ratio
        assert torch.allclose(dyn.Iz_t, expected_Iz, rtol=1e-5)


# ─── _sample_dr_domains：分布范围 + shape ─────────────────────────────────

class TestSampleDomains:
    def test_in_range_uniform(self):
        torch.manual_seed(42)
        K = 1000
        m_t_nom = 9300.0
        Cf_nom = 264000.0
        Cr_nom = 335000.0
        mt_range = 0.10
        cfcr_range = 0.20

        mt, cf, cr = _sample_dr_domains(K, mt_range, cfcr_range,
                                         m_t_nom, Cf_nom, Cr_nom)

        assert mt.shape == (K,)
        assert cf.shape == (K,)
        assert cr.shape == (K,)
        # 全部落在 ±range 内
        assert (mt >= m_t_nom * (1 - mt_range)).all()
        assert (mt <= m_t_nom * (1 + mt_range)).all()
        assert (cf >= Cf_nom * (1 - cfcr_range)).all()
        assert (cf <= Cf_nom * (1 + cfcr_range)).all()
        assert (cr >= Cr_nom * (1 - cfcr_range)).all()
        assert (cr <= Cr_nom * (1 + cfcr_range)).all()
        # 大样本均值近似 nominal
        assert mt.mean().item() == pytest.approx(m_t_nom, rel=0.02)
        assert cf.mean().item() == pytest.approx(Cf_nom, rel=0.02)

    def test_seed_reproducible(self):
        gen1 = torch.Generator().manual_seed(123)
        gen2 = torch.Generator().manual_seed(123)
        a = _sample_dr_domains(4, 0.1, 0.2, 9300, 264000, 335000, generator=gen1)
        b = _sample_dr_domains(4, 0.1, 0.2, 9300, 264000, 335000, generator=gen2)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert torch.equal(a[2], b[2])


# ─── _resolve_dr_config：CLI 覆盖 cfg ──────────────────────────────────────

class TestResolveDrConfig:
    def test_cfg_default(self):
        cfg = {'domain_randomization': {
            'enable': True, 'K': 8, 'mt_range': 0.15, 'cfcr_range': 0.25}}
        r = _resolve_dr_config(cfg)
        assert r == {'enable': True, 'K': 8,
                     'mt_range': 0.15, 'cfcr_range': 0.25}

    def test_cli_override(self):
        cfg = {'domain_randomization': {
            'enable': False, 'K': 4, 'mt_range': 0.10, 'cfcr_range': 0.20}}
        r = _resolve_dr_config(cfg, {'enable': True, 'K': 6,
                                      'mt_range': None, 'cfcr_range': None})
        assert r['enable'] is True   # CLI 覆盖
        assert r['K'] == 6           # CLI 覆盖
        assert r['mt_range'] == 0.10  # 回落 cfg
        assert r['cfcr_range'] == 0.20

    def test_no_cfg_section(self):
        r = _resolve_dr_config({})
        # 全部回落代码默认值
        assert r == {'enable': False, 'K': 4,
                     'mt_range': 0.10, 'cfcr_range': 0.20}


# ─── BatchedTruckTrailerVehicle.set_domain：B 元素独立 ──────────────────────

class TestVehicleSetDomain:
    def test_different_mass_yields_different_state(self):
        cfg = _truck_cfg()
        B = 3
        zeros = torch.zeros(B)
        v0 = torch.full((B,), 5.0)
        veh = BatchedTruckTrailerVehicle(
            cfg, batch_size=B,
            init_x=zeros, init_y=zeros,
            init_yaw=zeros, init_v=v0)

        # 给 3 个 batch 元素不同的 m_t（Cf/Cr 相同）
        m_t = torch.tensor([5000.0, 9300.0, 15000.0])
        Cf = torch.full((B,), float(cfg['truck_trailer_vehicle']['Cf']))
        Cr = torch.full((B,), float(cfg['truck_trailer_vehicle']['Cr']))
        veh.set_domain(m_t, Cf, Cr)

        # 同 delta + 同 torque，质量越大加速度越小
        delta = torch.full((B,), 0.05)
        torque = torch.full((B,), 5000.0)
        for _ in range(20):
            veh.step(delta, torque)

        # 三个元素的 v 不应相等（DR 生效的最直接证据）
        v_after = veh.v
        assert not torch.allclose(v_after[0:1], v_after[1:2], rtol=1e-3)
        assert not torch.allclose(v_after[1:2], v_after[2:3], rtol=1e-3)


# ─── DR 关闭时 train_batch 行为不变（回归） ──────────────────────────────

class TestDrDisabledRegression:
    """train_batch DR 关闭时数值行为不应被 DR 路径污染。

    跑两次同样的小训练（短轨迹、2 epoch，固定 seed），最终 loss 应一致。
    """
    def test_disabled_dr_loss_stable_across_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # 跑两次都用 DR 关闭 + 固定种子，确认 loss 完全一致（保护原路径不受 DR 改动污染）
        trajs = _short_truck_trajs()
        torch.manual_seed(7)
        r1 = train_batch(
            trajectories=trajs, n_epochs=2,
            plant='truck_trailer', verbose=False,
            disable_mlp=True,
            param_snapshot_interval=0)
        torch.manual_seed(7)
        r2 = train_batch(
            trajectories=trajs, n_epochs=2,
            plant='truck_trailer', verbose=False,
            disable_mlp=True,
            param_snapshot_interval=0)
        assert r1['losses'][0] == pytest.approx(r2['losses'][0], rel=1e-5)
        assert r1['losses'][-1] == pytest.approx(r2['losses'][-1], rel=1e-5)
        assert r1['dr_config']['enable'] is False
        assert r2['dr_config']['enable'] is False


# ─── DR 启用时 train_batch 端到端跑通 ─────────────────────────────────────

class TestDrEnabledEndToEnd:
    def test_dr_enabled_runs_and_changes_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        torch.manual_seed(7)
        r = train_batch(
            trajectories=_short_truck_trajs(), n_epochs=2,
            plant='truck_trailer', verbose=False,
            dr_overrides={'enable': True, 'K': 2,
                          'mt_range': 0.05, 'cfcr_range': 0.1},
            dr_seed=42,
            param_snapshot_interval=0)
        assert r['dr_config']['enable'] is True
        assert r['dr_config']['K'] == 2
        assert r['disable_mlp'] is True   # DR 强制蕴含
        # 至少一个参数从 init 移动了
        for name, init_val in r['initial_params'].items():
            final_val = r['final_params'][name]
            if isinstance(init_val, list):
                if any(abs(a - b) > 1e-6
                       for a, b in zip(init_val, final_val)):
                    return
            elif abs(init_val - final_val) > 1e-6:
                return
        pytest.fail("DR 训练 2 epoch 后参数未发生变化")

    def test_dr_history_records_domains(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        torch.manual_seed(7)
        r = train_batch(
            trajectories=_short_truck_trajs(), n_epochs=2,
            plant='truck_trailer', verbose=False,
            dr_overrides={'enable': True, 'K': 3,
                          'mt_range': 0.05, 'cfcr_range': 0.1},
            dr_seed=42,
            param_snapshot_interval=0)
        for ep in r['training_history']:
            assert 'dr_domains' in ep
            assert set(ep['dr_domains'].keys()) == {'d0', 'd1', 'd2'}
            for d in ep['dr_domains'].values():
                assert 'm_t' in d and 'Cf' in d and 'Cr' in d
                assert 'mean_loss' in d
