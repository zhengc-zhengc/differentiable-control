# sim/tests/test_torque_output.py
"""纵向扭矩输出层测试——验证公式与 C++ CalFinalTorque 对齐。"""
import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import load_config
from controller.lon import LonController


def _compute_expected_torque(
        acc_cmd, v, a_actual,
        *, m=9300.0, coef_cd=0.6, coef_rolling=0.013, coef_delta=1.05,
        air_density=1.2041, g=9.81, frontal_area=9.7,
        wheel_rolling_radius=0.5, kp=1000.0, deadzone=-0.05):
    """直接套公式，用于对比 controller 输出。"""
    F_air = 0.5 * coef_cd * air_density * frontal_area * v * v
    F_rolling = coef_rolling * m * g
    F_inertia = coef_delta * m * acc_cmd
    F_resist = F_air + F_rolling + F_inertia
    F_P = kp * (acc_cmd - a_actual)
    T_raw = (F_resist + F_P) * wheel_rolling_radius
    return T_raw if acc_cmd > deadzone else 0.0


class TestTorqueOutputLayer:

    def test_formula_v1_matches_expected(self):
        """v1 路径应与手算公式一致。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        T_sim = ctrl.compute_torque_wheel(acc_cmd=1.0, speed_mps=20.0, a_actual=0.5)
        T_ref = _compute_expected_torque(1.0, 20.0, 0.5)
        assert T_sim == pytest.approx(T_ref, rel=1e-5)

    def test_deadzone_outputs_zero(self):
        """acc_cmd 在死区下（< -0.05）应输出 0。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        T_sim = ctrl.compute_torque_wheel(
            acc_cmd=-0.1, speed_mps=10.0, a_actual=-0.1)
        assert T_sim == 0.0

    def test_deadzone_boundary_positive_side(self):
        """acc_cmd 略大于 -0.05 时应输出非零扭矩。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        T_sim = ctrl.compute_torque_wheel(
            acc_cmd=-0.04, speed_mps=10.0, a_actual=-0.04)
        # F_inertia 为负、F_P=0，但 F_rolling + F_air 主导为正
        assert T_sim > 0.0

    def test_differentiable_matches_v1(self):
        """可微路径数值应与 v1 一致（仅计算图不同）。"""
        cfg = load_config()
        ctrl_v1 = LonController(cfg, differentiable=False)
        ctrl_diff = LonController(cfg, differentiable=True)
        T_v1 = ctrl_v1.compute_torque_wheel(
            acc_cmd=0.8, speed_mps=15.0, a_actual=0.3)
        T_diff = ctrl_diff.compute_torque_wheel(
            acc_cmd=0.8, speed_mps=15.0, a_actual=0.3)
        assert T_diff.item() == pytest.approx(T_v1, rel=1e-5)

    def test_gradient_flows_to_acc_cmd(self):
        """可微路径梯度应能回传到 acc_cmd。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=True)
        acc = torch.tensor(1.0, requires_grad=True)
        v = torch.tensor(20.0)
        a_actual = torch.tensor(0.5)
        T = ctrl.compute_torque_wheel(acc, v, a_actual)
        T.backward()
        assert acc.grad is not None
        assert acc.grad.item() != 0.0

    def test_deadzone_mask_blocks_gradient_in_braking(self):
        """死区内（扭矩=0）梯度应为 0——mask.detach() 切断门控梯度。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=True)
        acc = torch.tensor(-0.1, requires_grad=True)
        v = torch.tensor(20.0)
        a_actual = torch.tensor(-0.1)
        T = ctrl.compute_torque_wheel(acc, v, a_actual)
        T.backward()
        # mask=0 时，T=0×T_raw，整个图梯度全乘 0
        assert acc.grad.item() == pytest.approx(0.0, abs=1e-9)

    def test_torque_scales_with_acceleration(self):
        """同速度下，acc_cmd 越大扭矩越大（单调性）。"""
        cfg = load_config()
        ctrl = LonController(cfg, differentiable=False)
        T_low = ctrl.compute_torque_wheel(
            acc_cmd=0.2, speed_mps=10.0, a_actual=0.2)
        T_high = ctrl.compute_torque_wheel(
            acc_cmd=1.0, speed_mps=10.0, a_actual=1.0)
        assert T_high > T_low
