# sim/tests/test_post_training.py
"""训练后自动化测试。"""
import os
import pytest
import sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _make_mock_train_result(saved_path=None):
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


class TestPlotLossCurves:
    def test_creates_file(self, tmp_path):
        from optim.post_training import plot_loss_curves
        result = _make_mock_train_result()
        path = plot_loss_curves(result['training_history'], str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('loss_curve.png')
        assert os.path.getsize(path) > 0


class TestPlotLossBreakdown:
    def test_creates_file(self, tmp_path):
        from optim.post_training import plot_loss_breakdown
        result = _make_mock_train_result()
        path = plot_loss_breakdown(result['training_history'],
                                   ['circle', 'sine'], str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('loss_breakdown.png')

    def test_single_trajectory(self, tmp_path):
        from optim.post_training import plot_loss_breakdown
        result = _make_mock_train_result()
        path = plot_loss_breakdown(result['training_history'],
                                   ['circle'], str(tmp_path))
        assert os.path.exists(path)


class TestPlotTrainingSummary:
    def test_creates_file(self, tmp_path):
        from optim.post_training import plot_training_summary
        result = _make_mock_train_result()
        comparison = {
            'circle': {
                'baseline': {'lat_rmse': 0.066, 'head_rmse': 0.009, 'lat_max': 0.18, 'head_max': 0.02},
                'tuned': {'lat_rmse': 0.064, 'head_rmse': 0.0088, 'lat_max': 0.17, 'head_max': 0.02},
                'delta_lat_pct': -3.0, 'delta_head_pct': -2.2,
            },
            'sine': {
                'baseline': {'lat_rmse': 0.788, 'head_rmse': 0.095, 'lat_max': 1.39, 'head_max': 0.15},
                'tuned': {'lat_rmse': 0.780, 'head_rmse': 0.094, 'lat_max': 1.38, 'head_max': 0.14},
                'delta_lat_pct': -1.0, 'delta_head_pct': -1.1,
            },
        }
        hyperparams = {'epochs': 10, 'lr': 1e-3, 'trajectories': ['circle', 'sine'],
                       'speed': 5.0, 'tbptt_k': 64, 'grad_clip': 10.0}
        path = plot_training_summary(result, comparison, hyperparams, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('training_summary.png')
        assert os.path.getsize(path) > 0


class TestPlotParameterChanges:
    def test_creates_file(self, tmp_path):
        from optim.post_training import plot_parameter_changes
        result = _make_mock_train_result()
        path = plot_parameter_changes(result, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith('parameter_changes.png')
        assert os.path.getsize(path) > 0

    def test_only_scalars(self, tmp_path):
        from optim.post_training import plot_parameter_changes
        result = _make_mock_train_result()
        # 去掉查找表参数
        del result['initial_params']['lat_ctrl.T2_y']
        del result['final_params']['lat_ctrl.T2_y']
        path = plot_parameter_changes(result, str(tmp_path))
        assert os.path.exists(path)

    def test_only_tables(self, tmp_path):
        from optim.post_training import plot_parameter_changes
        result = _make_mock_train_result()
        # 去掉标量参数
        del result['initial_params']['lon_ctrl.station_kp']
        del result['initial_params']['lon_ctrl.station_ki']
        del result['final_params']['lon_ctrl.station_kp']
        del result['final_params']['lon_ctrl.station_ki']
        path = plot_parameter_changes(result, str(tmp_path))
        assert os.path.exists(path)


class TestSaveExperimentLog:
    def test_creates_valid_yaml(self, tmp_path):
        from config import load_config, save_tuned_config
        from optim.post_training import save_experiment_log
        # 创建临时 tuned config
        cfg = load_config()
        saved = save_tuned_config(cfg, output_dir=str(tmp_path))
        result = _make_mock_train_result(saved_path=saved)
        hyperparams = {'epochs': 3, 'lr': 1e-3, 'trajectories': ['circle', 'sine']}
        path = save_experiment_log(result, {}, str(tmp_path), hyperparams, saved)
        assert os.path.exists(path)
        with open(path, 'r') as f:
            log = yaml.safe_load(f)
        assert 'commit' in log
        assert 'hyperparams' in log
        assert log['results']['initial_loss'] == pytest.approx(3.66)
        assert log['results']['final_loss'] == pytest.approx(3.60)

    def test_parameter_changes_recorded(self, tmp_path):
        from config import load_config, save_tuned_config
        from optim.post_training import save_experiment_log
        cfg = load_config()
        saved = save_tuned_config(cfg, output_dir=str(tmp_path))
        result = _make_mock_train_result(saved_path=saved)
        hyperparams = {'epochs': 3}
        path = save_experiment_log(result, {}, str(tmp_path), hyperparams, saved)
        with open(path, 'r') as f:
            log = yaml.safe_load(f)
        assert 'parameter_changes' in log
        assert 'lon_ctrl.station_kp' in log['parameter_changes']
        assert log['parameter_changes']['lon_ctrl.station_kp']['initial'] == pytest.approx(0.25)
