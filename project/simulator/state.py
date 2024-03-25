from dataclasses import dataclass
import collections
import copy
import types
import torch
import numpy as np
from typing import List
from project.domain.operation import Operation
from project.simulator.action import Action

@dataclass
class State:
    """状态实体
    Args
    -----
    batch_opr_features: 工序特征, shape -> (batch_size, n_oprs, opr_feat_dim)
    batch_station_features: 工位特征, shape -> (batch_size, n_stations, station_feat_dim)
    batch_edge_features: 工序-工位边特征, shape -> (batch_size, n_oprs, n_stations)
    baatch_opr_station: 工序-工位链接关系, shape -> (batch_size, n_oprs, n_stations)
    batch_num_opr: 记录任务的工序的个数, shape -> (batch_size, n_jobs)
    batch_opr_job: 记录工序的任务索引, shape -> (batch_size, n_oprs)
    batch_first_opr: 记录任务的首工序, shape -> (batch_size, n_jobs)
    batch_opr_next: 记录工序的后继工序, shape -> (batch_size, n_oprs, n_oprs)
    batch_opr_pre: 记录工序的前向工序, shape -> (batch_size, n_oprs, n_oprs)
    batch_mainline_flag: 记录工序是否主线的标识, shape -> (batch_size, n_oprs)
    batch_line_links: 记录主线与部装线工序链接情况, shape -> (batch_size, n_oprs, n_oprs)
    batch_subline_links: 记录所有部装线所链接的主线工序情况, shape -> (batch_size, n_oprs, n_oprs)
    batch_mainline_num_opr: 记录任务中主线工序的个数, shape -> (batch_size, n_jobs)
    batch_opr_line_index: 记录工序所在线(主线或部装线)的索引, shape -> (batch_size, n_oprs)
    batch_jobs: 记录每个批次的任务数, shape -> (batch_size,)
    params: dict
    """
    batch_opr_features: torch.Tensor
    batch_station_features: torch.Tensor
    batch_edge_features: torch.Tensor
    batch_opr_station: torch.Tensor
    batch_num_opr: torch.Tensor
    batch_opr_job: torch.Tensor
    batch_first_opr: torch.Tensor
    batch_opr_next: torch.Tensor
    batch_opr_pre: torch.Tensor
    batch_mainline_flag: torch.Tensor
    batch_line_links: torch.Tensor
    batch_subline_links: torch.Tensor
    batch_mainline_num_opr: torch.Tensor
    batch_opr_line_index: torch.Tensor
    batch_jobs: torch.Tensor
    params: dict

    def __post_init__(self):
        self._inject_params()
        #self.n_jobs = torch.max(self.batch_jobs)[0]
        # 记录每批工序总数 shape -> (batch_size, )
        self.n_total_opr = torch.sum(self.batch_num_opr, dim=-1).long()
        # 记录实例索引,每个实例包含多个任务(订单), shape -> (batch_size,)
        self.batch_indices = torch.arange(self.batch_size).long()
        self.ori_batch_indices = torch.arange(self.batch_size).long()
        # 记录时间索引, shape -> (batch_size,)
        self.batch_time = torch.zeros(self.batch_size)
        # 统计每个任务已调度的工序数
        self.n_scheduled_job_opr = torch.zeros(self.batch_size, self.n_jobs).long()
        # 统计已调度的工序数 shape -> (batch_size,)
        self.n_scheduled_opr = torch.zeros(self.batch_size,).long()
        # 记录任务的末工序 shape -> (batch_size, n_jobs)
        self.batch_last_opr = self.batch_first_opr + self.batch_num_opr - 1
        # 记录工序的依赖工序
        self.batch_opr_depencies = copy.deepcopy(self.batch_opr_pre)

        self.batch_opr_proctime = copy.deepcopy(self.batch_edge_features)

        # 记录工位剩余可分配工序
        self.batch_opr_station_remain = copy.deepcopy(self.batch_opr_station)

        self.batch_ori_opr_station = copy.deepcopy(self.batch_opr_station)

        # 记录合法的部分工序(针对单日最大约束)
        self.batch_valid_oprs = torch.full(size=(self.batch_size, self.batch_opr_job.size()[-1]), dtype=bool, fill_value=True)

        # 记录`NO_ACT`动作执行的次数
        self.batch_no_act = torch.zeros(self.batch_size,).long()

        # 记录各条线的第一道工序
        #print(self.batch_opr_pre.shape)
        #print((self.batch_opr_pre.sum(dim=-1).squeeze(-1)).shape)
        self.is_first = torch.where(self.batch_opr_pre.sum(dim=-1).squeeze(-1) == 0, True, False)
        #print(self.is_first.shape)
        #print(self.is_first)
        #raise Exception()
        self._build_mask()

        self._build_opr_schedule_state()
        self._build_station_schedule_state()

        self._build_cumsum_opr()

        self.batch_done = self.batch_mask_finish_job.all(dim=1)  # shape: (batch_size,)
        self.batch_makespan = torch.max(self.batch_opr_features[:, :, 4], dim=1)[0]

    def _inject_params(self):
        """注入配置参数
        """
        self.batch_size = self.params['batch_size']
        self.device = self.params['device']
        self.n_jobs = self.params['n_jobs']
        self.source = self.params.get('source', 'simulation')
        self.n_lines = self.params.get('n_lines', 1)
        self.exist_no_action = self.params.get('no_action', False)
        self.is_train = self.params.get('is_train', False)
        self.is_test = self.params.get('is_test', False)
        self.n_stations = self.params['n_stations']
        # 针对真实场景数据,每条线加入了一个虚拟工位
        if self.source in ('as',):
            self.n_stations += (self.n_jobs * self.n_lines)
        # 最后一个任务仅包含一个工序,表示不执行任务动作
        if self.exist_no_action:
            print('添加动作 `NO_ACTION`')
            self.n_jobs += 1
        # 是否存在部装线
        self.exist_subline = self.batch_line_links.any()
        self.opr_feat_dim = self.params['opr_feat_dim']
        self.station_feat_dim = self.params['station_feat_dim']
        self.modify_factor = (5 + 20) // 2

        self.open_no_act = torch.full(size=(self.batch_size, self.n_stations), dtype=torch.bool, fill_value=False)
        self.steps = torch.ones(size=(self.batch_size,))

    def _build_mask(self):
        """构建掩码
        """
        # 已排产的工序掩码, shape -> (batch_size, n_oprs)
        self.batch_mask_finish_opr = torch.full(size=(self.batch_size, max(self.n_total_opr).item()), dtype=torch.bool, fill_value=False)
        # 已完成的任务(订单)掩码, shape -> (batch_size, n_jobs)
        self.batch_mask_finish_job = torch.full(size=(self.batch_size, self.n_jobs), dtype=torch.bool, fill_value=False)
        # 正在处理工序的机器(工位)掩码, shape -> (batch_size, n_stations)
        self.batch_mask_busy_station = torch.full(size=(self.batch_size, self.n_stations), dtype=torch.bool, fill_value=False)
        # 正在处理的任务掩码, shape -> (batch_size, n_jobs)
        self.batch_mask_busy_job = torch.full(size=(self.batch_size, self.n_jobs), dtype=torch.bool, fill_value=False)

    def _build_opr_schedule_state(self):
        """构建工序或任务(订单)的调度状态信息
        Features:
            status: 工序状态, 1: 已排产 0: 未排产
            allocated machines: 已分配的工位(机器)
            start time: 工序开始时间
            end time: 工序完工时间
            job index: 所属任务索引
        """
        self.batch_opr_schedule = torch.zeros(size=(self.batch_size, max(self.batch_num_opr.sum(dim=1)), 5))
        self.batch_opr_schedule[..., 2] = self.batch_opr_features[..., 5]
        self.batch_opr_schedule[..., 3] = self.batch_opr_features[..., 5] + self.batch_opr_features[..., 2]

    def _build_station_schedule_state(self):
        """构建工位(机器)的调度状态信息
        Features:
            idle: 是否空闲, 1: 空闲, 0: 忙碌
            available time: 可开始时间
            utilization time: 已占产能时长
            job indices: 正在处理的任务索引 
        """
        self.batch_station_schedule = torch.zeros(size=(self.batch_size, self.n_stations, 4))
        self.batch_station_schedule[..., 0] = torch.ones(size=(self.batch_size, self.n_stations))
        self.batch_station_schedule[..., 3] -= 1

    def _build_cumsum_opr(self):
        """构建依赖工序积累情况
        """
        self.cumsum_opr = torch.zeros(*self.batch_opr_pre.size())
        # 判断各个工序是否存在前置工序, shape -> (batch_size, n_oprs)
        exist_dependencies = torch.sum(self.batch_opr_pre, dim=-1).squeeze(-1)
        for b in range(self.batch_size):
            each_exist_deps = exist_dependencies[b, ...].squeeze(0)
            num_oprs = self.batch_num_opr[b, ...].squeeze().cpu().numpy()
            n_line_oprs = []
            c = 0
            #print(num_oprs)
            for cnt in num_oprs:
                if cnt == 0:
                    break
                each_job_exist_deps = each_exist_deps[c:c+cnt]
                #print(each_job_exist_deps.shape)
                c += cnt
                # 当前实例各个主线及部装线的首工序, shape -> (n_lines,)
                indep_oprs = torch.where(each_job_exist_deps == 0)[0]
                #print(indep_oprs.shape)
                # 获取实例各个主线及部装线的工序个数, shape -> (n_lines,)
                n_line_opr = indep_oprs[1:] - indep_oprs[:-1]
                #print(n_line_opr.shape)
                n_line_oprs.append(torch.cat((n_line_opr, 
                                             torch.tensor([each_job_exist_deps.size(0)-indep_oprs[-1]]))).squeeze().reshape(-1,1))
            n_line_oprs = [item.reshape(-1) for item in n_line_oprs]
            n_line_oprs = torch.hstack(n_line_oprs)

            # shape -> (n_oprs, n_oprs)
            opr_pre = copy.deepcopy(self.batch_opr_pre[b, ...].squeeze(0))
            opr_pre[self.batch_line_links[b, ...].squeeze(0)] = 0

            s = 0
            for n in n_line_oprs:
                self.cumsum_opr[b, s:s+n, :] = opr_pre[s:s+n, :].cumsum(dim=0).float()
                s += n
    
    def __repr__(self):
        return f'opr feature shape -> {self.batch_opr_features.shape}, \
                    station feature shape -> {self.batch_station_features.shape} \
                    edge feature shape -> {self.batch_edge_features.shape}'