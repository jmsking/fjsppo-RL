from dataclasses import dataclass
import numpy as np
import torch
from project.simulator.state import State
from project.constraints._constraint import _Constraint
from project.constraints.filter_factory import register_filter

@register_filter('base_constraint')
class BaseConstraint(_Constraint):
    """基础约束下的动作选择
    """
    def __init__(self, eligible, state, memory=None):
        super().__init__(eligible, state, memory)

    def choose(self):
        """
        Returns
        -------
        eligible: torch.Tensor, shape -> (batch_size, n_oprs, n_stations)
            每批次的工序-工位对
        """
        # 用于弥补浮点数矩阵乘法带来的累计误差
        gap = 0.0
        # 获取工序最早可开始时间
        start_time = self.state.batch_opr_features[..., 5]
        #print(start_time)
        # 当前时间步
        time = self.state.batch_time.unsqueeze(-1)
        time = time.expand_as(start_time)
        # 工序可开始时间小于等于当前时间步,则为当前时刻合法的工序, shape -> (batch_size, n_oprs, 1)
        valid_oprs = torch.where(start_time <= time, True, False).unsqueeze(-1)
        # 筛选合法工序
        valid_oprs = valid_oprs.expand_as(self.eligible)
        self.eligible = torch.where(valid_oprs, self.eligible, False)
        # 得到空闲的工位, shape -> (batch_size, n_stations)
        #print(self.state.batch_mask_busy_station)
        idle_stations = ~self.state.batch_mask_busy_station
        #print(idle_stations.shape)
        #print(self.eligible.shape)
        #print(idle_stations[0])
        idle_stations = idle_stations.unsqueeze(1).expand_as(self.eligible)
        # shape -> (batch_size, n_oprs, n_stations)
        self.eligible = self.eligible & idle_stations
        return self.eligible
