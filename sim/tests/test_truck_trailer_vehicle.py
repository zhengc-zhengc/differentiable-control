# sim/tests/test_truck_trailer_vehicle.py
"""牵引车-挂车双体动力学模型测试。"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.truck_trailer_vehicle import TruckTrailerVehicle


# 与 truckdynamicmodel BASE_MODEL_PARAMS 一致
TT_PARAMS = {
    'm_t': 9300.0, 'Iz_t': 48639.0, 'L_t': 4.475, 'a_t': 3.8,
    'm_s_base': 15004.0, 'Iz_s_base': 96659.0, 'L_s': 8.0, 'c_s': 4.0,
    'Cf': 80000.0, 'Cr': 80000.0, 'Cs': 80000.0,
    'wheel_radius': 0.5, 'track_width': 1.8, 'steering_ratio': 16.39,
    'rho': 1.225, 'CdA_t': 5.82, 'CdA_s': 6.50, 'rolling_coeff': 0.006,
    'hitch_x': -0.331, 'hitch_y': 0.002, 'min_speed_mps': 0.5,
}

# 本地 checkpoint（v1：18D 输入 / 6D 输出 / 128×4）
_CHECKPOINT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'checkpoints',
    'truck_trailer_error_model.pth'))
_HAS_CHECKPOINT = os.path.exists(_CHECKPOINT)

# v2 checkpoint（14D 输入 / 9D 输出 / 64×3，含相对位姿残差）
_CHECKPOINT_V2 = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'checkpoints',
    'best_truck_trailer_error_model.pth'))
_HAS_CHECKPOINT_V2 = os.path.exists(_CHECKPOINT_V2)


class TestTruckTrailerVehicleNoCheckpoint:
    """无 checkpoint：纯 base 物理模型。"""

    def test_interface(self):
        car = TruckTrailerVehicle(TT_PARAMS, x=1.0, y=2.0, yaw=0.1, v=5.0,
                                  dt=0.02)
        assert isinstance(car.x, torch.Tensor)
        assert isinstance(car.y, torch.Tensor)
        assert isinstance(car.yaw, torch.Tensor)
        assert isinstance(car.v, torch.Tensor)
        assert hasattr(car, 'speed_kph')
        assert hasattr(car, 'yaw_deg')
        assert hasattr(car, 'step')
        assert hasattr(car, 'detach_state')
        assert hasattr(car, 'trailer_state')

    def test_rear_axle_init_consistent(self):
        """初始化传后轴坐标，反向取 x/y 应能取回（容许 dt=0 误差）。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=10.0, y=20.0, yaw=0.5, v=0.0,
                                  dt=0.02)
        assert car.x.item() == pytest.approx(10.0, abs=1e-4)
        assert car.y.item() == pytest.approx(20.0, abs=1e-4)
        assert car.yaw.item() == pytest.approx(0.5, abs=1e-6)

    def test_straight_line(self):
        """零转角+零扭矩，直行：y≈0、yaw≈0，x 前进但因阻力衰减。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02)
        for _ in range(500):
            car.step(delta=0.0, torque_wheel=0.0)
        assert car.x.item() > 30.0
        assert abs(car.y.item()) < 1.0
        assert abs(car.yaw.item()) < 0.01

    def test_acceleration(self):
        """正车轮扭矩 → 速度增大。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=2.0, dt=0.02)
        v0 = car.v.item()
        # 9300+15004=24304 kg, a≈0.5 m/s² ~ T_wheel = m·a·R = 24304·0.5·0.5 ≈ 6076 N·m
        for _ in range(50):
            car.step(delta=0.0, torque_wheel=6000.0)
        assert car.v.item() > v0

    def test_steering_causes_lateral(self):
        """方向盘转角 → y 偏移。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=8.0, dt=0.02)
        delta_front = 0.05  # 前轮转角 (rad)
        for _ in range(150):
            car.step(delta=delta_front, torque_wheel=0.0)
        assert abs(car.y.item()) > 0.2

    def test_no_trailer_mode(self):
        """trailer_mass < 1.0 应进入无挂车模式（不崩溃）。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
                                  trailer_mass_kg=0.0)
        for _ in range(50):
            car.step(delta=0.0, torque_wheel=0.0)
        assert car.x.item() > 0

    def test_detach_state(self):
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                                  differentiable=True)
        torque = torch.tensor(800.0, requires_grad=True)
        car.step(delta=0.0, torque_wheel=torque)
        car.detach_state()
        assert not car.x.requires_grad
        assert not car.v.requires_grad

    def test_gradient_flows(self):
        """differentiable 模式下梯度应能回传到 torque_wheel。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                                  differentiable=True)
        torque = torch.tensor(2000.0, requires_grad=True)
        car.step(delta=0.0, torque_wheel=torque)
        car.v.backward()
        assert torque.grad is not None
        assert torque.grad.item() != 0.0


@pytest.mark.skipif(not _HAS_CHECKPOINT,
                    reason="truckdynamicmodel checkpoint 不存在")
class TestTruckTrailerVehicleWithCheckpoint:
    """带 MLP 残差 checkpoint。"""

    def test_load_checkpoint(self):
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
                                  checkpoint_path=_CHECKPOINT)
        assert car._mlp is not None

    def test_mlp_correction_changes_trajectory(self):
        """带 MLP 与不带 MLP 的轨迹应有差异。"""
        car_base = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0,
                                       dt=0.02)
        car_hybrid = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0,
                                         dt=0.02, checkpoint_path=_CHECKPOINT)
        for _ in range(100):
            car_base.step(delta=0.02, torque_wheel=0.0)
            car_hybrid.step(delta=0.02, torque_wheel=0.0)
        diff = (abs(car_base.x.item() - car_hybrid.x.item())
                + abs(car_base.y.item() - car_hybrid.y.item()))
        assert diff > 0.001

    def test_gradient_flows_with_mlp(self):
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                                  differentiable=True,
                                  checkpoint_path=_CHECKPOINT)
        torque = torch.tensor(2000.0, requires_grad=True)
        car.step(delta=0.0, torque_wheel=torque)
        car.v.backward()
        assert torque.grad is not None


@pytest.mark.skipif(not _HAS_CHECKPOINT_V2,
                    reason="v2 checkpoint 不存在")
class TestTruckTrailerVehicleV2Checkpoint:
    """v2 checkpoint：14D 输入、9D 输出（6 速度残差 + 3 相对位姿残差）、64×3 隐层。"""

    def test_load_v2_checkpoint(self):
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
                                  checkpoint_path=_CHECKPOINT_V2)
        assert car._mlp is not None
        assert car._mlp_input_dim == 14
        assert car._mlp_output_dim == 9
        # 首层权重宽度应为 64（新架构 hidden_dim）
        first_linear = next(m for m in car._mlp.network
                            if isinstance(m, torch.nn.Linear))
        assert first_linear.out_features == 64

    def test_v2_feature_tensor_shape(self):
        """v2 特征构造应产出 14D。"""
        from model.truck_trailer_vehicle import (
            build_mlp_input_feature_tensor_v2,
        )
        state = torch.zeros(1, 12)
        control = torch.tensor([[0.1, 0.0, 0.0, 300.0, 300.0]])
        mass = torch.tensor([[0.0]])
        dt = torch.tensor([[0.02]])
        features = build_mlp_input_feature_tensor_v2(state, control, mass, dt)
        assert features.shape == (1, 14)

    def test_v2_mlp_correction_runs(self):
        """v2 MLP 端到端不崩溃，位姿前进。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
                                  checkpoint_path=_CHECKPOINT_V2,
                                  trailer_mass_kg=15004.0)
        for _ in range(50):
            car.step(delta=0.01, torque_wheel=500.0)
        assert car.x.item() > 1.0

    def test_v2_no_trailer_invariant(self):
        """trailer_mass=0 时 MLP 修正不应破坏 '挂车态=牵引车态'。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=10.0, dt=0.02,
                                  checkpoint_path=_CHECKPOINT_V2,
                                  trailer_mass_kg=0.0)
        for i in range(100):
            car.step(delta=0.02 * (i % 5 - 2) / 2.0, torque_wheel=500.0)
        ts = car.trailer_state
        # 挂车 6D 状态应严格等于牵引车 6D 状态
        for i in range(6):
            assert ts[i].item() == pytest.approx(car._state[i].item(), abs=1e-6)

    def test_v2_gradient_flows(self):
        """v2 differentiable 模式下梯度可以回传到扭矩。"""
        car = TruckTrailerVehicle(TT_PARAMS, x=0, y=0, yaw=0, v=5.0, dt=0.02,
                                  differentiable=True,
                                  checkpoint_path=_CHECKPOINT_V2)
        torque = torch.tensor(2000.0, requires_grad=True)
        car.step(delta=0.0, torque_wheel=torque)
        car.v.backward()
        assert torque.grad is not None
        assert torque.grad.item() != 0.0
