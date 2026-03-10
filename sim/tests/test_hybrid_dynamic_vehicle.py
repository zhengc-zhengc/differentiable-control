# sim/tests/test_hybrid_dynamic_vehicle.py
"""混合动力学车辆模型测试。"""
import math
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.hybrid_dynamic_vehicle import (
    HybridDynamicVehicle, MLPErrorModel, _reconstruct_full_error)

# 与 plant 仓库一致的标定参数
HYBRID_PARAMS = {
    'mass': 2440.0,
    'Iz': 9564.8,
    'lf': 1.354,
    'lr': 1.446,
    'wheel_radius': 0.329,
    'drag_coeff': 0.558,
    'frontal_area': 5.903,
    'air_density': 1.206,
    'rolling_coeff': 0.0065,
    'corner_stiff_f': 56000.0,
    'corner_stiff_r': 56000.0,
    'tire_friction_mu': 0.85,
    'track_width': 1.725,
    'steer_ratio': 16.39,
    'reverse_sign_speed_mps': 0.5,
}

# plant checkpoint 路径
_PLANT_CHECKPOINT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'plant', 'best_error_model_from_carsim.pth'))

_HAS_CHECKPOINT = os.path.exists(_PLANT_CHECKPOINT)


class TestHybridDynamicVehicleNoCheckpoint:
    """无 checkpoint 时的基础测试（纯 Base Euler 模型）。"""

    def test_interface_has_required_attributes(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=1.0, y=2.0, yaw=0.1, v=5.0, dt=0.02)
        assert isinstance(car.x, torch.Tensor)
        assert isinstance(car.y, torch.Tensor)
        assert isinstance(car.yaw, torch.Tensor)
        assert isinstance(car.v, torch.Tensor)
        assert hasattr(car, 'speed_kph')
        assert hasattr(car, 'yaw_deg')
        assert hasattr(car, 'step')
        assert hasattr(car, 'detach_state')

    def test_straight_line_no_mlp(self):
        """无 MLP 时，直行应保持 y~0、yaw~0。"""
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        for _ in range(500):
            car.step(delta=0.0, acc=0.0)
        assert car.x.item() > 50.0
        assert abs(car.y.item()) < 0.5
        assert abs(car.yaw.item()) < 0.01

    def test_acceleration_no_mlp(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=1.0, dt=0.02)
        v0 = car.v.item()
        for _ in range(50):
            car.step(delta=0.0, acc=1.0)
        assert car.v.item() > v0

    def test_steering_no_mlp(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02)
        for _ in range(200):
            car.step(delta=0.05, acc=0.0)
        assert abs(car.y.item()) > 0.5

    def test_properties(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        assert car.speed_kph.item() == pytest.approx(36.0, abs=0.1)
        assert car.yaw_deg.item() == pytest.approx(0.0, abs=0.01)

    def test_detach_state(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
            differentiable=True)
        car.step(delta=0.0, acc=torch.tensor(1.0, requires_grad=True))
        car.detach_state()
        assert not car.x.requires_grad
        assert not car.v.requires_grad

    def test_gradient_flows_no_mlp(self):
        """无 MLP 时梯度应能回传到 acc。"""
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
            differentiable=True)
        acc = torch.tensor(1.0, requires_grad=True)
        car.step(delta=0.0, acc=acc)
        car.v.backward()
        assert acc.grad is not None
        assert acc.grad.item() != 0.0


class TestMLPErrorModel:
    def test_output_shape(self):
        mlp = MLPErrorModel(input_dim=10, output_dim=3)
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 3)

    def test_zero_init(self):
        """输出层零初始化 -> 初始输出应接近零。"""
        mlp = MLPErrorModel(input_dim=10, output_dim=3)
        x = torch.randn(1, 10)
        y = mlp(x)
        assert y.abs().max().item() < 1e-6


class TestReconstructFullError:
    def test_output_shape(self):
        motion_err = torch.zeros(2, 3)
        base_next = torch.zeros(2, 6)
        result = _reconstruct_full_error(motion_err, base_next, dt=0.02)
        assert result.shape == (2, 6)

    def test_zero_motion_gives_zero_full(self):
        motion_err = torch.zeros(1, 3)
        base_next = torch.tensor([[10.0, 5.0, 0.5, 8.0, 0.1, 0.01]])
        result = _reconstruct_full_error(motion_err, base_next, dt=0.02)
        assert result.abs().max().item() < 1e-10

    def test_vx_error_aligned_with_yaw(self):
        """yaw=0 时, vx 误差应只影响 x 方向。"""
        motion_err = torch.tensor([[1.0, 0.0, 0.0]])
        base_next = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])
        result = _reconstruct_full_error(motion_err, base_next, dt=0.02)
        # dx = vx_err * dt = 1.0 * 0.02 = 0.02
        assert result[0, 0].item() == pytest.approx(0.02, abs=1e-6)
        # dy = 0
        assert result[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        # dvx = 1.0 (直接传递)
        assert result[0, 3].item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not _HAS_CHECKPOINT,
                    reason="plant checkpoint 不存在")
class TestHybridDynamicVehicleWithCheckpoint:
    """加载 checkpoint 后的集成测试。"""

    def test_load_checkpoint(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
            checkpoint_path=_PLANT_CHECKPOINT)
        assert car._mlp is not None
        assert car._feature_mean is not None
        assert car._feature_scale is not None

    def test_straight_line_with_mlp(self):
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
            checkpoint_path=_PLANT_CHECKPOINT)
        for _ in range(500):
            car.step(delta=0.0, acc=0.0)
        # 即使有 MLP 修正，直行也应大致保持直行
        assert car.x.item() > 30.0
        assert abs(car.y.item()) < 5.0

    def test_gradient_flows_with_mlp(self):
        """MLP 权重冻结，但梯度仍能通过 MLP 计算图回传到 acc。"""
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
            differentiable=True,
            checkpoint_path=_PLANT_CHECKPOINT)
        acc = torch.tensor(1.0, requires_grad=True)
        car.step(delta=0.0, acc=acc)
        car.v.backward()
        assert acc.grad is not None
        assert acc.grad.item() != 0.0

    def test_mlp_weights_frozen(self):
        """MLP 参数不应有梯度。"""
        car = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
            differentiable=True,
            checkpoint_path=_PLANT_CHECKPOINT)
        for p in car._mlp.parameters():
            assert not p.requires_grad

    def test_mlp_correction_nonzero(self):
        """转向时 MLP 修正应非零。"""
        car_base = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        car_hybrid = HybridDynamicVehicle(
            HYBRID_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
            checkpoint_path=_PLANT_CHECKPOINT)

        for _ in range(100):
            car_base.step(delta=0.02, acc=0.0)
            car_hybrid.step(delta=0.02, acc=0.0)

        # 两者轨迹应有差异（MLP 修正生效）
        diff_x = abs(car_base.x.item() - car_hybrid.x.item())
        diff_y = abs(car_base.y.item() - car_hybrid.y.item())
        assert diff_x + diff_y > 0.01
