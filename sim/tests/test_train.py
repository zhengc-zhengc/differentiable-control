# sim/tests/test_train.py
"""训练 pipeline 测试：DiffControllerParams + tracking_loss + train()（truck_trailer，降强度）。"""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optim.train import DiffControllerParams, tracking_loss, train


PLANT = 'truck_trailer'
SHORT_LEN = 15.0


class TestDiffControllerParams:
    def test_has_parameters(self):
        """封装模块应包含可优化参数：横向 T2-T6 y 值 + 纵向 7 个 PID 标量。"""
        params = DiffControllerParams()
        n_params = sum(p.numel() for p in params.parameters())
        assert n_params > 30, f"参数数量不足: {n_params}"

    def test_to_config_dict_structure(self):
        params = DiffControllerParams()
        cfg = params.to_config_dict()
        assert 'vehicle' in cfg
        assert 'lat_truck' in cfg and 'lon' in cfg
        lat = cfg['lat_truck']
        for k in ('kLh', 'T1_max_theta_deg', 'T8_slip_param'):
            assert k in lat
        lon = cfg['lon']
        for k in ('station_kp', 'L5_rate_gain'):
            assert k in lon

    def test_to_config_dict_all_python_types(self):
        """导出的配置全部为 Python 原生类型（float/list），不含 tensor。"""
        params = DiffControllerParams()
        cfg = params.to_config_dict()
        lat = cfg['lat_truck']
        assert isinstance(lat['kLh'], float)
        assert isinstance(lat['T1_max_theta_deg'], list)
        assert isinstance(lat['T1_max_theta_deg'][0][0], float)
        assert isinstance(cfg['lon']['station_kp'], float)

    def test_parameter_groups(self):
        """参数应可分为 table_y 组和 other 组。"""
        params = DiffControllerParams()
        table_params = [n for n, _ in params.named_parameters() if '_y' in n]
        other_params = [n for n, _ in params.named_parameters() if '_y' not in n]
        assert len(table_params) > 0
        assert len(other_params) > 0


class TestTrackingLoss:
    def test_zero_error_zero_loss(self):
        history = [{'lateral_error': torch.tensor(0.0),
                    'heading_error': torch.tensor(0.0),
                    'v': torch.tensor(5.0),
                    'steer': torch.tensor(0.0),
                    'acc': torch.tensor(0.0)} for _ in range(50)]
        loss = tracking_loss(history, ref_speed=5.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_error_increases_loss(self):
        """横向误差/速度误差增大都应推高 loss。"""
        def make(lat, v):
            return [{'lateral_error': torch.tensor(lat),
                     'heading_error': torch.tensor(0.0),
                     'v': torch.tensor(v),
                     'steer': torch.tensor(0.0),
                     'acc': torch.tensor(0.0)} for _ in range(30)]
        assert tracking_loss(make(1.0, 5.0), ref_speed=5.0).item() > \
               tracking_loss(make(0.1, 5.0), ref_speed=5.0).item()
        assert tracking_loss(make(0.0, 3.0), ref_speed=5.0).item() > \
               tracking_loss(make(0.0, 5.0), ref_speed=5.0).item()

    def test_return_details_has_loss_components(self):
        history = [{'lateral_error': torch.tensor(0.1),
                    'heading_error': torch.tensor(0.05),
                    'v': torch.tensor(4.5),
                    'steer': torch.tensor(float(i)),
                    'acc': torch.tensor(0.1)} for i in range(20)]
        loss, details = tracking_loss(history, ref_speed=5.0, return_details=True)
        for key in ('loss_lat', 'loss_head', 'loss_speed',
                    'loss_steer_rate', 'loss_acc_rate'):
            assert key in details
        component_sum = sum(details[k] for k in
                            ('loss_lat', 'loss_head', 'loss_speed',
                             'loss_steer_rate', 'loss_acc_rate'))
        assert abs(component_sum - loss.item()) < 1e-4

    def test_loss_is_differentiable(self):
        lat_err = torch.tensor(1.0, requires_grad=True)
        history = [{'lateral_error': lat_err,
                    'heading_error': torch.tensor(0.0),
                    'v': torch.tensor(5.0),
                    'steer': torch.tensor(0.0),
                    'acc': torch.tensor(0.0)} for _ in range(10)]
        loss = tracking_loss(history, ref_speed=5.0)
        loss.backward()
        assert lat_err.grad is not None


class TestTrain:
    def test_pipeline_runs_and_saves(self):
        """训练 pipeline 跑 1 epoch × 1 轨迹 × 短仿真，验证保存的 yaml 可加载。"""
        from config import load_config
        result = train(
            trajectories=['lane_change'],
            n_epochs=1,
            lr=1e-2,
            sim_length=SHORT_LEN,
            verbose=False,
            plant=PLANT,
        )
        assert len(result['losses']) == 1
        assert result['saved_path'] is not None
        assert 'training_history' in result
        for loss_val in result['losses']:
            assert not torch.isnan(torch.tensor(loss_val))
            assert not torch.isinf(torch.tensor(loss_val))
        cfg = load_config(result['saved_path'])
        assert 'vehicle' in cfg and 'lat_truck' in cfg and 'lon' in cfg
        os.remove(result['saved_path'])
