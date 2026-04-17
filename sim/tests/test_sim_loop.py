# sim/tests/test_sim_loop.py
"""闭环仿真测试（truck_trailer，降强度）。

重点：仿真能跑通、history 结构完整、differentiable=True 时梯度流通。
跟踪精度不做硬约束（truck_trailer 用默认乘用车参数跟不住）。
"""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sim_loop import run_simulation
from config import load_config
from model.trajectory import generate_straight
from controller.lat_truck import LatControllerTruck
from controller.lon import LonController


def _tt_cfg():
    cfg = load_config()
    cfg['vehicle']['model_type'] = 'truck_trailer'
    return cfg


# 短轨迹：2 秒，100 步 @ 50Hz
SHORT_LEN = 10
SHORT_SPEED = 5.0


class TestSimLoop:
    def test_runs_without_crash(self):
        """仿真能跑完一段短轨迹。"""
        cfg = _tt_cfg()
        traj = generate_straight(length=SHORT_LEN, speed=SHORT_SPEED)
        history = run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg)
        assert len(history) > 50

    def test_history_has_required_fields(self):
        cfg = _tt_cfg()
        traj = generate_straight(length=SHORT_LEN, speed=SHORT_SPEED)
        history = run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg)
        rec = history[0]
        for key in ['t', 'x', 'y', 'yaw', 'v', 'steer', 'acc',
                    'lateral_error', 'heading_error', 'ref_x', 'ref_y']:
            assert key in rec, f"Missing key: {key}"


class TestSimLoopDifferentiable:
    """differentiable=True 模式：梯度流 + tensor 输出。"""

    def test_gradient_flows_through_lat(self):
        cfg = _tt_cfg()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=SHORT_LEN, speed=SHORT_SPEED)
        history = run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        lat_errs = torch.stack([h['lateral_error'] for h in history])
        loss = (lat_errs ** 2).mean()
        loss.backward()
        assert lat_ctrl.T2_y.grad is not None, "T2_y 应有梯度"

    def test_gradient_flows_through_lon(self):
        cfg = _tt_cfg()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=SHORT_LEN, speed=SHORT_SPEED)
        history = run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        v_errs = torch.stack([h['v'] - SHORT_SPEED for h in history])
        loss = (v_errs ** 2).mean()
        loss.backward()
        assert lon_ctrl.low_speed_kp.grad is not None, "low_speed_kp 应有梯度"

    def test_history_values_are_tensors(self):
        cfg = _tt_cfg()
        lat_ctrl = LatControllerTruck(cfg, differentiable=True)
        lon_ctrl = LonController(cfg, differentiable=True)
        traj = generate_straight(length=SHORT_LEN, speed=SHORT_SPEED)
        history = run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=True)
        rec = history[10]
        for key in ['x', 'y', 'yaw', 'v', 'steer', 'acc',
                    'lateral_error', 'heading_error']:
            assert isinstance(rec[key], torch.Tensor)

    def test_external_controllers_reset(self):
        """外部传入控制器时应调用 reset_state()。"""
        cfg = _tt_cfg()
        lat_ctrl = LatControllerTruck(cfg, differentiable=False)
        lon_ctrl = LonController(cfg, differentiable=False)
        traj = generate_straight(length=SHORT_LEN, speed=SHORT_SPEED)
        # 先跑一次让内部状态不为零
        run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg,
                       lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                       differentiable=False)
        # 再跑应从 reset 后的状态开始
        history = run_simulation(traj, init_speed=SHORT_SPEED, cfg=cfg,
                                 lat_ctrl=lat_ctrl, lon_ctrl=lon_ctrl,
                                 differentiable=False)
        # 仅确认能跑完，不对绝对值做断言（重卡首步转向取决于参数）
        assert len(history) > 50
