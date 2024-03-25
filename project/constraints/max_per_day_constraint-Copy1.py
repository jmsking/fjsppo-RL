from dataclasses import dataclass
import numpy as np
import torch
from project.simulator.action import Action
from project.common.memory import Memory
from project.constraints._constraint import _Constraint
from project.constraints.filter_factory import register_filter
from project.common.constant import SECOND_PER_DAY

#@register_filter('max_per_day_constraint')
class MaxPerDayConstraint(_Constraint):
    """单日最大约束下的动作选择
    """
    def __init__(self, eligible, state, memory):
        super().__init__(eligible, state, memory)
        self.manager = self.memory.manager
        self.per_day_counter = self.manager.per_day_counter
        self.per_day_jobs = self.manager.per_day_jobs
        self.batch_time = self.state.batch_time
        self.batch_size = self.state.batch_size
        

    def choose(self):
        if len(self.per_day_counter) == 0:
            return self.eligible
        # TODO 性能优化,修改为矩阵形式计算?
        for b in range(self.batch_size):
            each_ins_eligible = self.eligible[b, ...]
            each_ins_per_day_counter = self.per_day_counter[b]
            each_ins_time = self.batch_time[b].cpu().item()
            #TODO 与日历时间对应
            day = int(each_ins_time / SECOND_PER_DAY)
            # 当天各个工位上已排产的任务数, shape -> (n_stations,)
            if day not in each_ins_per_day_counter:
                continue
            counter = each_ins_per_day_counter[day]
            #print(f'Counter for day {day} -> {counter}')
            #print(f'Jobs for day {day} -> {self.manager.per_day_jobs[b][day]}')
            #单日最大数量
            #TODO 进行配置
            station_filters = torch.where(counter < 3, True, False)
            station_filters = station_filters.expand_as(each_ins_eligible)
            each_ins_eligible = torch.where(station_filters, each_ins_eligible, False)
            #print(station_filters.any())
            #print(each_ins_eligible.any())
            # 当天任务数达到最大值后,需要跨天
            if (~station_filters).all():
            #if (~station_filters).any():
                #print(each_ins_eligible.any())
                self.state.batch_station_schedule[b, :, 1] = (torch.div(self.batch_time[b], SECOND_PER_DAY, rounding_mode='floor') + 1) * SECOND_PER_DAY
                #print(self.state.batch_station_schedule[b, :, 1])
            else:
                self.eligible[b, ...] = each_ins_eligible

        return self.eligible
