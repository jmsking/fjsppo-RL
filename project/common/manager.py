import collections
import torch

from project.simulator.state import State
from project.simulator.action import Action
from project.common.constant import SECOND_PER_DAY

class Manager:
    """管理器
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # 单天任务统计
        self.per_day_counter = []
        # 记录任务
        self.per_day_jobs = []

    def statistic(self, state: State, action: Action):
        self._count_oprs_per_day(state, action)

    def _count_oprs_per_day(self, state: State, action: Action):
        """统计每天每个工位排产的工序数
        """
        # 每个实例工序分配的工位, shape -> (batch_size,)
        stations = action.opr_station_pair[1, :]
        # 每个实例的当前排产的任务(订单)，shape -> (batch_size,)
        jobs = action.opr_station_pair[2, :]
        # 当前的时间步, shape -> (batch_size,)
        curr_time_step = state.batch_time
        batch_size = curr_time_step.size(0)
        batch_indices = state.batch_indices.cpu()
        if len(self.per_day_counter) == 0:
            self.per_day_counter = [{} for _ in range(batch_size)]
            self.per_day_jobs = [collections.defaultdict(set) for _ in range(batch_size)]
        # 累加每个工位某一天处理的任务数
        for idx, b in enumerate(batch_indices):
            b = int(b.item())
            job = jobs[idx].cpu().item()
            station = stations[idx].cpu().item()
            # 忽略`NO_ACT`
            if state.exist_no_action and job == state.n_jobs-1:
                continue
            day = int(curr_time_step[b].cpu().item() / SECOND_PER_DAY)
            
            if day not in self.per_day_counter[b]:
                self.per_day_counter[b][day] = torch.zeros(size=(state.n_stations,)).long()
                #self.per_day_counter[b][day][station] += 1
            # 同一天处理的任务不同
            #elif job not in self.per_day_jobs[b][day]:
            #elif job not in self.per_day_counter[b[]]
            self.per_day_counter[b][day][station] += 1
            self.per_day_jobs[b][day].add(job)
        #print(f'单天排产任务数 -> {self.per_day_jobs}')
