# sim/tests/test_post_training.py
"""训练后可视化/日志测试（纯 mock，不跑仿真）。"""
import os
import pytest
import sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _mock_result(saved_path=None):
    """构造模拟的 train() 返回值。"""
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
        'saved_path': saved_path,
        'params': None,
    }


class TestPlotters:
    def test_loss_curve(self, tmp_path):
        from optim.post_training import plot_loss_curves
        path = plot_loss_curves(_mock_result()['training_history'], str(tmp_path))
        assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_loss_breakdown(self, tmp_path):
        from optim.post_training import plot_loss_breakdown
        path = plot_loss_breakdown(
            _mock_result()['training_history'],
            ['circle', 'sine'], str(tmp_path))
        assert os.path.exists(path)

    def test_training_summary(self, tmp_path):
        from optim.post_training import plot_training_summary
        comparison = {
            'circle': {
                'baseline': {'lat_rmse': 0.066, 'head_rmse': 0.009, 'lat_max': 0.18, 'head_max': 0.02},
                'tuned': {'lat_rmse': 0.064, 'head_rmse': 0.0088, 'lat_max': 0.17, 'head_max': 0.02},
                'delta_lat_pct': -3.0, 'delta_head_pct': -2.2,
            },
        }
        hyperparams = {'epochs': 10, 'lr': 1e-3, 'trajectories': ['circle'],
                       'speed': 5.0, 'tbptt_k': 64, 'grad_clip': 10.0}
        path = plot_training_summary(_mock_result(), comparison, hyperparams,
                                     str(tmp_path))
        assert os.path.exists(path)

    def test_parameter_changes(self, tmp_path):
        from optim.post_training import plot_parameter_changes
        path = plot_parameter_changes(_mock_result(), str(tmp_path))
        assert os.path.exists(path)


class TestExperimentLog:
    def test_yaml_roundtrip(self, tmp_path):
        """日志 YAML 应含关键字段，loss 与 mock 数据一致。"""
        from config import load_config, save_tuned_config
        from optim.post_training import save_experiment_log
        cfg = load_config()
        saved = save_tuned_config(cfg, output_dir=str(tmp_path))
        result = _mock_result(saved_path=saved)
        hyperparams = {'epochs': 3, 'lr': 1e-3, 'trajectories': ['circle', 'sine']}
        path = save_experiment_log(result, {}, str(tmp_path), hyperparams, saved)
        with open(path, 'r') as f:
            log = yaml.safe_load(f)
        assert 'commit' in log
        assert 'hyperparams' in log
        assert log['results']['initial_loss'] == pytest.approx(3.66)
        assert log['results']['final_loss'] == pytest.approx(3.60)
        assert 'parameter_changes' in log
        assert 'lon_ctrl.station_kp' in log['parameter_changes']
