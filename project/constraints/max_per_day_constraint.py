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
            each_ins_per_day_counter = self.per_day_jobs[b]
            each_ins_time = self.batch_time[b].cpu().item()
            #TODO 与日历时间对应
            day = int(each_ins_time / SECOND_PER_DAY)
            # 当天各个工位上已排产的任务数, shape -> (n_stations,)
            if day not in each_ins_per_day_counter:
                continue
            if len(each_ins_per_day_counter[day]) < 3:
                continue
            # 工序所属任务 shape -> (n_oprs,)
            opr_job = self.state.batch_opr_job[b]
            counter = each_ins_per_day_counter[day]
            # 得到所属当前任务的所有工序
            sel_oprs = torch.isin(opr_job, torch.tensor(list(counter)+[self.state.n_jobs-1]))
            self.state.batch_valid_oprs[b] = sel_oprs
            sel_opr_indices = torch.where(sel_oprs)[0]
            scheduled = torch.where(self.state.batch_opr_schedule[b, sel_opr_indices[:-1], 0] == 1, True, False)
            if scheduled.all():
                # 当前任务集合中的工序都已排完,则进行跨天
                self.state.batch_station_schedule[b, :, 1] = (torch.div(self.batch_time[b], SECOND_PER_DAY, rounding_mode='floor') + 1) * SECOND_PER_DAY
                self.eligible[:] = self.eligible & False
                #print(self.state.batch_station_schedule[b, :, 1])
                #raise Exception()
                continue
            # 先排完当前任务集合
            opr_filters = sel_oprs.unsqueeze(-1).expand_as(each_ins_eligible)
            each_ins_eligible = torch.where(opr_filters, each_ins_eligible, False)
            self.eligible[b, ...] = each_ins_eligible
            #print(f'Counter for day {day} -> {counter}')
            #print(f'Jobs for day {day} -> {self.manager.per_day_jobs[b][day]}')
            #单日最大数量
            #TODO 进行配置
            #station_filters = torch.where(counter < 3, True, False)
            
            #print(station_filters.any())
            #print(each_ins_eligible.any())

        return self.eligible
