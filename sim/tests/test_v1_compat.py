# sim/tests/test_v1_compat.py
"""V1 兼容性验证测试。
确保 differentiable=False 模式的行为与 V1 完全一致。
覆盖横向控制器、纵向控制器、闭环仿真三个层级。
"""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from model.trajectory import (generate_straight, generate_circle,
                              generate_sine, generate_combined, TrajectoryAnalyzer)
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController
from sim_loop import run_simulation


# ── 横向控制器 V1 兼容性 ──────────────────────────────────


class TestV1CompatLat:
    """LatControllerTruck V1 兼容性：differentiable=False 应与 V1 行为一致。"""

    @pytest.fixture
    def setup(self):
        cfg = load_config()
        ctrl = LatControllerTruck(cfg, differentiable=False)
        return cfg, ctrl

    def test_straight_no_steer(self, setup):
        """在直线上、无偏差时，转向输出应接近 0。"""
        cfg, ctrl = setup
        traj = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(traj)
        steer, k_cur, k_near, k_far, _, _ = ctrl.compute(
            x=0.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True, dt=0.02)
        assert isinstance(steer, float), "V1 路径应返回 float"
        assert abs(steer) < 5.0, f"直线无偏差应转向接近 0，实际={steer:.3f}"

    def test_lateral_offset_corrects(self, setup):
        """有横向偏差时应产生修正转向。"""
        cfg, ctrl = setup
        traj = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(traj)
        steer, _, _, _, _, _ = ctrl.compute(
            x=5.0, y=2.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True, dt=0.02)
        # 正偏差 y=2 → 应向右修正（负转向角）
        assert steer < 0, f"正横向偏差应产生负转向，实际={steer:.3f}"

    def test_disable_returns_feedback(self, setup):
        """ctrl_enable=False 时应返回 steer_feedback 原值。"""
        cfg, ctrl = setup
        traj = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(traj)
        steer, k_cur, k_near, k_far, _, _ = ctrl.compute(
            x=0.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            yawrate=0.0, steer_feedback=15.5,
            analyzer=analyzer, ctrl_enable=False, dt=0.02)
        assert steer == pytest.approx(15.5, abs=0.01)
        assert k_cur == 0.0
        assert k_near == 0.0
        assert k_far == 0.0

    def test_curvature_output(self, setup):
        """圆弧轨迹上应输出非零曲率。"""
        cfg, ctrl = setup
        traj = generate_circle(radius=50.0, speed=5.0,
                               arc_angle=math.pi / 2)
        analyzer = TrajectoryAnalyzer(traj)
        steer, k_cur, k_near, k_far, _, _ = ctrl.compute(
            x=0.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            yawrate=0.0, steer_feedback=0.0,
            analyzer=analyzer, ctrl_enable=True, dt=0.02)
        assert abs(k_cur) > 0 or abs(k_near) > 0 or abs(k_far) > 0, \
            "圆弧轨迹应有非零曲率输出"

    def test_multi_step_rate_limiting(self, setup):
        """连续调用应受速率限制约束。"""
        cfg, ctrl = setup
        traj = generate_straight(length=100, speed=10.0)
        analyzer = TrajectoryAnalyzer(traj)
        steers = []
        for i in range(10):
            steer_fb = steers[-1] if steers else 0.0
            steer, _, _, _, _, _ = ctrl.compute(
                x=5.0, y=3.0, yaw_deg=0.0, speed_kph=36.0,
                yawrate=0.0, steer_feedback=steer_fb,
                analyzer=analyzer, ctrl_enable=True, dt=0.02)
            steers.append(steer)
        # 相邻步之间变化不应超过 rate_limit_total * dt
        max_rate = ctrl.rate_limit_total * 0.02
        for i in range(1, len(steers)):
            delta = abs(steers[i] - steers[i-1])
            assert delta <= max_rate + 0.01, \
                f"步 {i} 转向变化 {delta:.3f} 超过速率限制 {max_rate:.3f}"


# ── 纵向控制器 V1 兼容性 ──────────────────────────────────


class TestV1CompatLon:
    """LonController V1 兼容性：differentiable=False 应与 V1 行为一致。"""

    @pytest.fixture
    def setup(self):
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        traj = generate_straight(length=200, speed=10.0)
        analyzer = TrajectoryAnalyzer(traj)
        return cfg, ctrl, analyzer

    def test_on_track_no_correction(self, setup):
        """车辆在轨迹上、速度匹配时，加速度应接近 0。"""
        cfg, ctrl, analyzer = setup
        acc = ctrl.compute(
            x=10.0, y=0.0, yaw_deg=0.0, speed_kph=36.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=1.0,
            ctrl_enable=True, ctrl_first_active=True, dt=0.02)
        assert isinstance(acc, float), "V1 路径应返回 float"
        assert abs(acc) < 2.0, f"速度匹配时加速度应小，实际={acc:.3f}"

    def test_too_slow_accelerates(self, setup):
        """车速低于参考速度时应加速。"""
        cfg, ctrl, analyzer = setup
        # 参考速度 10 m/s (36 kph)，当前 5 m/s (18 kph)
        accs = []
        for step in range(5):
            acc = ctrl.compute(
                x=5.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
                accel_mps2=0.0, curvature_far=0.0,
                analyzer=analyzer, t_now=step * 0.02,
                ctrl_enable=True, ctrl_first_active=(step == 0), dt=0.02)
            accs.append(acc)
        # 至少后面几步应正加速
        assert any(a > 0 for a in accs), \
            f"车速偏低时应加速，accs={accs}"

    def test_too_fast_decelerates(self, setup):
        """车速高于参考速度时应减速。"""
        cfg, ctrl, analyzer = setup
        # 参考速度 10 m/s (36 kph)，当前 20 m/s (72 kph)
        accs = []
        for step in range(10):
            acc = ctrl.compute(
                x=10.0, y=0.0, yaw_deg=0.0, speed_kph=72.0,
                accel_mps2=0.0, curvature_far=0.0,
                analyzer=analyzer, t_now=step * 0.02,
                ctrl_enable=True, ctrl_first_active=(step == 0), dt=0.02)
            accs.append(acc)
        assert any(a < 0 for a in accs), \
            f"车速偏高时应减速，accs={accs}"

    def test_acc_within_limits(self, setup):
        """加速度应在配置的上下限表范围内。"""
        cfg, ctrl, analyzer = setup
        for step in range(20):
            acc = ctrl.compute(
                x=step * 0.2, y=0.0, yaw_deg=0.0, speed_kph=18.0,
                accel_mps2=0.0, curvature_far=0.0,
                analyzer=analyzer, t_now=step * 0.02,
                ctrl_enable=True, ctrl_first_active=(step == 0), dt=0.02)
            assert -5.0 < acc < 5.0, \
                f"加速度 {acc:.3f} 超出合理范围"

    def test_first_active_resets(self, setup):
        """ctrl_first_active=True 应重置 PID 状态。"""
        cfg, ctrl, analyzer = setup
        # 先跑几步积累状态
        for step in range(5):
            ctrl.compute(
                x=5.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
                accel_mps2=0.0, curvature_far=0.0,
                analyzer=analyzer, t_now=step * 0.02,
                ctrl_enable=True, ctrl_first_active=(step == 0), dt=0.02)
        # 用 first_active 重置
        acc_reset = ctrl.compute(
            x=5.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=0.0,
            ctrl_enable=True, ctrl_first_active=True, dt=0.02)
        # 重置后应与首步行为接近
        ctrl2 = LonController(cfg, differentiable=False)
        acc_fresh = ctrl2.compute(
            x=5.0, y=0.0, yaw_deg=0.0, speed_kph=18.0,
            accel_mps2=0.0, curvature_far=0.0,
            analyzer=analyzer, t_now=0.0,
            ctrl_enable=True, ctrl_first_active=True, dt=0.02)
        # 注意 acc_out_prev 不会被 first_active 重置，但 PID 会
        # 所以可能不完全相同，但应同号
        assert (acc_reset > 0) == (acc_fresh > 0), \
            f"重置后应与新建控制器同号: {acc_reset:.3f} vs {acc_fresh:.3f}"


# ── 闭环仿真 V1 兼容性 ───────────────────────────────────


class TestV1CompatSimLoop:
    """完整闭环仿真 V1 兼容性。"""

    def test_straight_tracks(self):
        """直线轨迹跟踪：后段横向误差 < 1m。"""
        traj = generate_straight(length=200, speed=10.0)
        history = run_simulation(traj, init_speed=10.0, differentiable=False)
        n_last = 100
        for rec in history[-n_last:]:
            assert abs(rec['lateral_error']) < 1.0, \
                f"直线跟踪横向误差过大: {rec['lateral_error']:.3f}"

    def test_circle_tracks(self):
        """圆弧轨迹跟踪：后半段横向误差 < 5m。"""
        traj = generate_circle(radius=50.0, speed=5.0,
                               arc_angle=math.pi / 2)
        history = run_simulation(traj, init_speed=5.0, differentiable=False)
        n = len(history)
        for rec in history[n // 2:]:
            assert abs(rec['lateral_error']) < 5.0, \
                f"圆弧跟踪横向误差过大: {rec['lateral_error']:.3f}"

    def test_combined_tracks(self):
        """组合轨迹（直线+弯道+直线）跟踪：后段误差可控。"""
        traj = generate_combined(speed=5.0, seg3_length=50.0)
        history = run_simulation(traj, init_speed=5.0, differentiable=False)
        # 检查最后 2 秒
        n_last = min(100, len(history) // 4)
        max_err = max(abs(rec['lateral_error']) for rec in history[-n_last:])
        assert max_err < 8.0, \
            f"组合轨迹后段最大横向误差 {max_err:.3f} > 8m"

    def test_history_values_are_float(self):
        """differentiable=False 时 history 中应为 float（非 tensor）。"""
        traj = generate_straight(length=50, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, differentiable=False)
        rec = history[10]
        for key in ['x', 'y', 'yaw', 'v', 'lateral_error', 'heading_error']:
            val = rec[key]
            assert isinstance(val, (int, float)), \
                f"V1 模式 history['{key}'] 应为 float，实际为 {type(val)}"

    def test_speed_converges(self):
        """速度应逐渐收敛到参考速度。"""
        traj = generate_straight(length=300, speed=10.0)
        # 从静止开始
        history = run_simulation(traj, init_speed=0.0, differentiable=False)
        # 最后 2 秒速度应接近 10 m/s
        n_last = 100
        for rec in history[-n_last:]:
            assert abs(rec['v'] - 10.0) < 3.0, \
                f"速度未收敛: v={rec['v']:.3f}，目标=10.0"

    def test_explicit_false_same_as_default(self):
        """显式 differentiable=False 应与不传参（默认）一致。"""
        traj = generate_straight(length=100, speed=5.0)
        h_default = run_simulation(traj, init_speed=5.0)
        h_explicit = run_simulation(traj, init_speed=5.0, differentiable=False)
        assert len(h_default) == len(h_explicit)
        for i in range(len(h_default)):
            for key in ['x', 'y', 'yaw', 'v', 'lateral_error']:
                v1 = h_default[i][key]
                v2 = h_explicit[i][key]
                if isinstance(v1, torch.Tensor):
                    v1 = v1.item()
                if isinstance(v2, torch.Tensor):
                    v2 = v2.item()
                assert abs(v1 - v2) < 1e-6, \
                    f"步 {i} 键 '{key}' 不一致: {v1} vs {v2}"

    def test_external_controllers_v1(self):
        """外部传入 differentiable=False 控制器应与内部创建一致。"""
        cfg = load_config()
        lat_ctrl = LatControllerTruck(cfg, differentiable=False)
        lon_ctrl = LonController(cfg, differentiable=False)
        traj = generate_straight(length=100, speed=5.0)
        h_external = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                    lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                    differentiable=False)
        h_internal = run_simulation(traj, init_speed=5.0, cfg=cfg,
                                    differentiable=False)
        assert len(h_external) == len(h_internal)
        for i in range(len(h_internal)):
            for key in ['x', 'y', 'v']:
                v1 = h_external[i][key]
                v2 = h_internal[i][key]
                if isinstance(v1, torch.Tensor):
                    v1 = v1.item()
                if isinstance(v2, torch.Tensor):
                    v2 = v2.item()
                assert abs(v1 - v2) < 1e-6, \
                    f"步 {i} 键 '{key}' 不一致: {v1} vs {v2}"

    def test_sine_bounded(self):
        """正弦轨迹仿真不发散。"""
        traj = generate_sine(amplitude=3.0, wavelength=50.0,
                             n_waves=2, speed=5.0)
        history = run_simulation(traj, init_speed=5.0, differentiable=False)
        for rec in history:
            assert abs(rec['lateral_error']) < 10.0, \
                f"正弦轨迹横向误差过大: {rec['lateral_error']:.3f}"
            assert abs(rec['v']) < 30.0, \
                f"速度失控: {rec['v']:.3f}"
