# sim/tests/test_train.py
"""训练 pipeline 测试：DiffControllerParams, tracking_loss, train()。"""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optim.train import DiffControllerParams, tracking_loss, train


class TestDiffControllerParams:
    def test_has_parameters(self):
        """封装模块应包含可优化参数：横向 T2-T6 y 值 + 纵向 7 个 PID 标量。"""
        params = DiffControllerParams()
        n_params = sum(p.numel() for p in params.parameters())
        # 5 张表 y 值 + 7 个标量，总数约 30-50
        assert n_params > 30, f"参数数量不足: {n_params}"

    def test_to_config_dict(self):
        """to_config_dict 应返回完整的配置字典。"""
        params = DiffControllerParams()
        cfg = params.to_config_dict()
        assert 'vehicle' in cfg
        assert 'lat_truck' in cfg
        assert 'lon' in cfg

    def test_to_config_dict_preserves_structure(self):
        """导出的配置应保留与 default.yaml 相同的键结构。"""
        params = DiffControllerParams()
        cfg = params.to_config_dict()
        lat = cfg['lat_truck']
        assert 'kLh' in lat
        assert 'T1_max_theta_deg' in lat
        assert 'T8_slip_param' in lat
        lon = cfg['lon']
        assert 'station_kp' in lon
        assert 'L5_rate_gain' in lon

    def test_to_config_dict_all_python_types(self):
        """导出的配置应全部为 Python 原生类型（float/list），不含 tensor。"""
        params = DiffControllerParams()
        cfg = params.to_config_dict()
        lat = cfg['lat_truck']
        assert isinstance(lat['kLh'], float)
        assert isinstance(lat['T1_max_theta_deg'], list)
        assert isinstance(lat['T1_max_theta_deg'][0], list)
        assert isinstance(lat['T1_max_theta_deg'][0][0], float)
        lon = cfg['lon']
        assert isinstance(lon['station_kp'], float)

    def test_contains_both_controllers(self):
        """应同时包含横向和纵向控制器。"""
        params = DiffControllerParams()
        assert hasattr(params, 'lat_ctrl')
        assert hasattr(params, 'lon_ctrl')
        assert isinstance(params.lat_ctrl, torch.nn.Module)
        assert isinstance(params.lon_ctrl, torch.nn.Module)

    def test_parameter_groups(self):
        """参数应可分为 table_y 组和 other 组。"""
        params = DiffControllerParams()
        table_params = []
        other_params = []
        for name, p in params.named_parameters():
            if '_y' in name:
                table_params.append(name)
            else:
                other_params.append(name)
        assert len(table_params) > 0, "应有查找表参数"
        assert len(other_params) > 0, "应有非查找表参数"


class TestTrackingLoss:
    def test_zero_error_zero_loss(self):
        """零误差时 loss 应为零。"""
        history = []
        for _ in range(100):
            history.append({
                'lateral_error': torch.tensor(0.0),
                'heading_error': torch.tensor(0.0),
                'v': torch.tensor(5.0),
                'steer': torch.tensor(0.0),
                'acc': torch.tensor(0.0),
            })
        loss = tracking_loss(history, ref_speed=5.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_lateral_error_increases_loss(self):
        """横向误差增大应使 loss 增大。"""
        def make_history(lat_err):
            return [{'lateral_error': torch.tensor(lat_err),
                     'heading_error': torch.tensor(0.0),
                     'v': torch.tensor(5.0),
                     'steer': torch.tensor(0.0),
                     'acc': torch.tensor(0.0)} for _ in range(50)]
        loss_small = tracking_loss(make_history(0.1), ref_speed=5.0)
        loss_large = tracking_loss(make_history(1.0), ref_speed=5.0)
        assert loss_large.item() > loss_small.item()

    def test_speed_error_increases_loss(self):
        """速度误差应使 loss 增大。"""
        def make_history(v):
            return [{'lateral_error': torch.tensor(0.0),
                     'heading_error': torch.tensor(0.0),
                     'v': torch.tensor(v),
                     'steer': torch.tensor(0.0),
                     'acc': torch.tensor(0.0)} for _ in range(50)]
        loss_match = tracking_loss(make_history(5.0), ref_speed=5.0)
        loss_off = tracking_loss(make_history(3.0), ref_speed=5.0)
        assert loss_off.item() > loss_match.item()

    def test_steer_rate_penalty(self):
        """转向变化率应计入 loss。"""
        history = []
        for i in range(50):
            history.append({
                'lateral_error': torch.tensor(0.0),
                'heading_error': torch.tensor(0.0),
                'v': torch.tensor(5.0),
                'steer': torch.tensor(float(i % 2) * 10.0),  # 交替 0/10
                'acc': torch.tensor(0.0),
            })
        loss = tracking_loss(history, ref_speed=5.0,
                             w_lat=0, w_head=0, w_speed=0,
                             w_steer_rate=1.0, w_acc_rate=0)
        assert loss.item() > 0.0

    def test_return_details_has_loss_components(self):
        """return_details 应包含各 loss 分项，且分项之和等于总 loss。"""
        history = [{'lateral_error': torch.tensor(0.1),
                    'heading_error': torch.tensor(0.05),
                    'v': torch.tensor(4.5),
                    'steer': torch.tensor(float(i)),
                    'acc': torch.tensor(0.1)} for i in range(20)]
        loss, details = tracking_loss(history, ref_speed=5.0, return_details=True)
        for key in ['loss_lat', 'loss_head', 'loss_speed', 'loss_steer_rate', 'loss_acc_rate']:
            assert key in details, f"缺少 {key}"
        component_sum = sum(details[k] for k in ['loss_lat', 'loss_head', 'loss_speed',
                                                  'loss_steer_rate', 'loss_acc_rate'])
        assert abs(component_sum - loss.item()) < 1e-4

    def test_loss_is_differentiable(self):
        """loss 应支持 backward。"""
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
    def test_pipeline_runs(self):
        """训练 pipeline 应能运行（短仿真 + 少 epoch）。"""
        result = train(
            trajectories=['lane_change'],
            n_epochs=2,
            lr=1e-2,
            sim_length=20.0,
            verbose=False,
        )
        assert len(result['losses']) == 2
        assert result['saved_path'] is not None
        assert result['params'] is not None
        assert 'training_history' in result
        assert len(result['training_history']) == 2
        assert 'per_trajectory' in result['training_history'][0]
        assert 'initial_params' in result
        assert 'final_params' in result

    def test_loss_finite(self):
        """训练 loss 应为有限数。"""
        result = train(
            trajectories=['lane_change'],
            n_epochs=3,
            lr=1e-3,
            sim_length=30.0,
            verbose=False,
        )
        for loss_val in result['losses']:
            assert not torch.isnan(torch.tensor(loss_val)), "loss 不应为 NaN"
            assert not torch.isinf(torch.tensor(loss_val)), "loss 不应为 Inf"

    def test_saved_config_loadable(self):
        """保存的配置文件应可加载。"""
        from config import load_config
        result = train(
            trajectories=['lane_change'],
            n_epochs=2,
            lr=1e-2,
            sim_length=20.0,
            verbose=False,
        )
        cfg = load_config(result['saved_path'])
        assert 'vehicle' in cfg
        assert 'lat_truck' in cfg
        assert 'lon' in cfg
        # 清理
        os.remove(result['saved_path'])

    def test_multi_trajectory(self):
        """多轨迹类型训练应能正常运行。"""
        result = train(
            trajectories=['lane_change', 's_curve'],
            n_epochs=2,
            lr=1e-3,
            sim_length=20.0,
            verbose=False,
        )
        assert len(result['losses']) == 2

    def test_multi_speed_trajectory(self):
        """多速度段训练应能正常运行（类型自动展开到全速度段）。"""
        result = train(
            trajectories=['lane_change'],
            n_epochs=2,
            lr=1e-3,
            sim_length=40.0,
            verbose=False,
        )
        assert len(result['losses']) == 2
        # 6 个速度段，per_trajectory 应各有明细
        hist = result['training_history'][0]
        assert 'lane_change_18kph' in hist['per_trajectory']
        assert 'lane_change_35kph' in hist['per_trajectory']
