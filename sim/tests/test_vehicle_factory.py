# sim/tests/test_vehicle_factory.py
"""车辆工厂测试（只覆盖 truck_trailer 分支 + 错误分支）。"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.truck_trailer_vehicle import TruckTrailerVehicle
from model.vehicle_factory import create_vehicle
from config import load_config


def _tt_cfg():
    cfg = load_config()
    cfg['vehicle']['model_type'] = 'truck_trailer'
    return cfg


class TestVehicleFactory:
    def test_creates_truck_trailer(self):
        """显式指定 truck_trailer 应返回 TruckTrailerVehicle。"""
        car = create_vehicle(_tt_cfg(), x=0, y=0, yaw=0, v=5.0, dt=0.02)
        assert isinstance(car, TruckTrailerVehicle)

    def test_truck_trailer_interface(self):
        """暴露统一接口：step(delta, torque_wheel) + 状态属性。"""
        car = create_vehicle(_tt_cfg(), x=0, y=0, yaw=0, v=5.0, dt=0.02)
        car.step(delta=0.0, torque_wheel=0.0)
        for attr in ('x', 'y', 'yaw', 'v', 'speed_kph', 'yaw_deg',
                     'detach_state'):
            assert hasattr(car, attr)

    def test_invalid_model_type_raises(self):
        cfg = load_config()
        cfg['vehicle']['model_type'] = 'invalid'
        with pytest.raises(ValueError, match='model_type'):
            create_vehicle(cfg, x=0, y=0, yaw=0, v=5.0, dt=0.02)

    def test_differentiable_forwarded(self):
        car = create_vehicle(_tt_cfg(), x=0, y=0, yaw=0, v=5.0, dt=0.02,
                             differentiable=True)
        assert car.differentiable is True
