# sim/model/vehicle_factory.py
"""车辆模型工厂：根据配置创建 BicycleModel / DynamicVehicle / HybridDynamicVehicle。"""
import os

from model.vehicle import BicycleModel
from model.dynamic_vehicle import DynamicVehicle
from model.hybrid_dynamic_vehicle import HybridDynamicVehicle


def _resolve_checkpoint_path(rel_path):
    """将 checkpoint 相对路径解析为绝对路径（相对于 sim/ 目录）。"""
    if not rel_path or os.path.isabs(rel_path):
        return rel_path
    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(sim_dir, rel_path))


def create_vehicle(cfg, x=0.0, y=0.0, yaw=0.0, v=0.0,
                   dt=0.02, differentiable=False):
    """根据 cfg['vehicle']['model_type'] 创建车辆模型。

    Args:
        cfg: 完整配置字典
        x, y, yaw, v: 初始状态
        dt: 仿真步长
        differentiable: 是否启用可微模式

    Returns:
        BicycleModel / DynamicVehicle / HybridDynamicVehicle 实例（接口一致）
    """
    model_type = cfg['vehicle'].get('model_type', 'kinematic')

    if model_type == 'kinematic':
        veh = cfg['vehicle']
        return BicycleModel(
            wheelbase=veh['wheelbase'], x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable)

    elif model_type == 'dynamic':
        dyn_params = cfg['dynamic_vehicle']
        return DynamicVehicle(
            params=dyn_params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable)

    elif model_type == 'hybrid_dynamic':
        hyb_params = cfg['hybrid_dynamic_vehicle']
        checkpoint = _resolve_checkpoint_path(
            hyb_params.get('checkpoint_path', ''))
        return HybridDynamicVehicle(
            params=hyb_params, x=x, y=y, yaw=yaw, v=v,
            dt=dt, differentiable=differentiable,
            checkpoint_path=checkpoint or None)

    else:
        raise ValueError(
            f"未知 vehicle.model_type: '{model_type}'，"
            f"支持: 'kinematic', 'dynamic', 'hybrid_dynamic'")
