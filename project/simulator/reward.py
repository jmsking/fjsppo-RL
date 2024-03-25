from dataclasses import dataclass
import torch
from project.simulator.state import State
from project.simulator.action import Action
from project.common.task_type import TaskType

@dataclass
class Reward:
    """某个状态下的奖励值
    Args
    ------
    pre_state: 前一个状态实体
    state: 当前状态实体
    action: 采取的动作
    """
    pre_state: State
    state: State
    action: Action

    def __post_init__(self):
        self.alpha1 = 0.2
        self.alpha2 = 0.5
        self.alpha3 = 0.6
        
    @property
    def value(self):
        """得到该状态下的奖励值
        """
        makespan_reward = self.optimize_makespan()
        #print(makespan_reward[0])
        opr_link_interval = self.optimize_opr_link_interval()
        job_priority = self.optimize_job_priority()
        no_action_penalty = self.optimize_no_action()
        station_reward = self.optimize_machine_busyness()
        #print(no_action_penalty[0])
        #print(station_reward[0])
        #total_reward = self.alpha1*makespan_reward - self.alpha2*opr_link_interval - self.alpha3*job_priority
        #total_reward += no_action_penalty
        #print(makespan_reward, job_priority)
        total_reward = makespan_reward + no_action_penalty
        #print(makespan_reward)
        #print(no_action_penalty)
        return total_reward

    def optimize_makespan(self):
        """优化目标: 最小化最大任务完工时长, 最大值为0
        """
        batch_makespan = torch.max(self.pre_state.batch_opr_features[:, :, 4], dim=1)[0]
        c_max = torch.max(self.state.batch_opr_features[:, :, 4], dim=1)[0]
        #print(batch_makespan[0])
        #print(c_max[0])
        batch_reward = batch_makespan - c_max
        return batch_reward
    
    def optimize_machine_busyness(self):
        """优化目标: 最大化每个时刻的机器繁忙度
        """
        # 获取当前时刻繁忙的工位数, shape -> (batch_size, n_stations)
        station_status = self.state.batch_station_schedule[..., 0].squeeze(-1)
        busy_cnt = torch.where(station_status == 0, 1, 0)
        # shape -> (batch_size, )
        busy_cnt = busy_cnt.sum(axis=-1).squeeze(-1)
        return busy_cnt




    def optimize_opr_link_interval(self):
        """优化目标: 最小化相邻工序最大间隔时长
        """
        batch_indices = self.pre_state.batch_indices
        oprs = self.action.opr_station_pair[0, :]
        # 获取工序的前置工序(依赖工序)的完工时间
        #TODO 考虑部装线
        dep_oprs = torch.nonzero(self.state.batch_opr_pre[batch_indices, oprs, :]).squeeze(-1)
        # 某个实例中可能正处理的是首工序
        if dep_oprs.size(0) == 0:
            return 0
        indices = dep_oprs[:, 0]
        batch_indices = batch_indices[indices]
        dep_oprs = dep_oprs[:, 1]
        # 获取当前工序的实际开始时间
        start_times = self.state.batch_opr_schedule[batch_indices, oprs[indices], 2].squeeze(-1)
        # 获取前置工序的实际完工时间
        end_times = self.state.batch_opr_schedule[batch_indices, dep_oprs, 3].squeeze(-1)
        # shape -> (batch_size,)
        max_interval = start_times - end_times
        interval = torch.zeros(*self.state.batch_time.size())
        interval[batch_indices] = max_interval
        return interval

    def optimize_job_priority(self):
        """优化目标: 最小化最大任务实际持续时长(从实际开始时间到实际结束时间的间隔)
        """
        job_first_oprs = self.state.batch_first_opr
        job_features = []
        for b in range(self.state.batch_size):
            job_features.append(self.state.batch_opr_features[b].index_select(0, job_first_oprs[b]).unsqueeze(0))
        job_features = torch.cat(job_features, dim=0)
        #print(job_first_oprs)
        #print(self.state.batch_opr_features)
        #print(job_features)
        #raise Exception()
        # 获取任务工序最早开始时间, shape -> (batch_size, n_jobs)
        start_time = job_features[..., 5].squeeze(-1)
        # 获取任务工序最晚完工时间, shape -> (batch_size, n_jobs)
        end_time = job_features[..., 4].squeeze(-1)
        # 获取任务间隔时长, shape -> (batch_size, n_jobs)
        interval_time = end_time - start_time
        # 获取最大间隔时长, shape -> (batch_size,)
        max_interval_time = torch.max(interval_time, dim=1).values
        return max_interval_time

    def optimize_no_action(self):
        """优化目标: 工序中间不允许采用NO_ACTION
        """
        specified_rewards = torch.zeros((self.pre_state.batch_size,))
        if not self.pre_state.exist_no_action:
            return specified_rewards
        # 当前处理的工序, shape -> (batch_size,)
        oprs = self.action.opr_station_pair[0, :]
        n_oprs = self.pre_state.n_total_opr[self.pre_state.batch_indices]
        # 不采用动作的实例掩码
        mask_instances = torch.where(oprs == n_oprs-1, True, False)
        #_, n_oprs, _ = self.state.batch_opr_station.size()
        # 过滤有动作的实例
        #mask_instances = torch.where(oprs == n_oprs-1, True, False)
        # 当前所有实例都未采取 `NO_ACTION`
        if (~mask_instances).all():
            return specified_rewards
        # 获取采用`NO_ACT`的实例
        batch_indices = self.pre_state.batch_indices[mask_instances]
        # 如果实例中存在其他已排产工序,则加大惩罚值(保证`NO_ACTION`要么在最开始执行,要么不执行), shape -> (sub_batch_size,)
        """is_scheduled = self.pre_state.batch_opr_schedule[batch_indices, :, 0].sum(dim=1).squeeze()
        is_scheduled = torch.where(is_scheduled >= 1, True, False)
        scheduled_batch_indices = batch_indices[is_scheduled]
        # `NO_ACTION`在最开始执行,不进行惩罚
        if scheduled_batch_indices.size(0) == 0:
            # 给予一定奖励值
            #busy_cnt = self.optimize_machine_busyness()
            #specified_rewards[batch_indices] = 0
            return specified_rewards"""
        # 在训练阶段,对于已完成的批次后续执行的`NO_ACT`不奖励
        mask = torch.where(self.state.batch_done[batch_indices], True, False)
        # 在其他位置执行`NO_ACTION`给予奖励值
        _ind = batch_indices[~mask]
        gamma = 100.0
        specified_rewards[_ind] = gamma * self.pre_state.steps[_ind] * 0.01
        return specified_rewards




