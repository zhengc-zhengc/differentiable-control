# sim/tests/test_health_check.py
"""health_check 模块测试。"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestCheckBaselinePerformance:
    def test_returns_four_scenarios(self):
        from health_check import check_baseline_performance
        results = check_baseline_performance()
        assert len(results) == 4

    def test_metrics_reasonable(self):
        from health_check import check_baseline_performance
        results = check_baseline_performance()
        for r in results:
            assert 'lat_rmse' in r
            assert 'head_rmse' in r
            assert 'lat_max' in r
            assert r['lat_rmse'] >= 0
            assert r['lat_rmse'] < 1.0, f"{r['name']} lat_rmse 异常: {r['lat_rmse']}"


class TestCheckGradientHealth:
    def test_returns_gradient_info(self):
        from health_check import check_gradient_health
        grad_info, total_norm, loss = check_gradient_health(
            trajectories=['lane_change'])
        assert len(grad_info) > 0
        assert total_norm >= 0
        assert loss > 0

    def test_some_params_have_gradient(self):
        from health_check import check_gradient_health
        grad_info, _, _ = check_gradient_health(
            trajectories=['lane_change'])
        n_ok = sum(1 for g in grad_info if g['status'] == 'OK')
        assert n_ok > 0, "应至少有部分参数有非零梯度"

    def test_gradient_info_fields(self):
        from health_check import check_gradient_health
        grad_info, _, _ = check_gradient_health(
            trajectories=['lane_change'])
        for g in grad_info:
            assert 'name' in g
            assert 'status' in g
            assert g['status'] in ('OK', 'WARN_ZERO', 'WARN_SMALL', 'ERROR')
            assert 'grad_norm' in g
