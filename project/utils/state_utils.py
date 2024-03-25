import copy
import time
import math
import numpy as np
import torch
import collections
from functools import partial
from project.simulator.action import Action
from project.simulator.state import State
from project.common.memory import Memory
from project.constraints.rule_filter import RuleFilter

class StateUtils:
    """环境状态工具类
    """
    @staticmethod
    def build_features(batch_oprs: list, batch_opr_stations: list, batch_opr_links: list, **params):
        """构造输入特征
        Features:
            batch_opr_features: 工序特征 shape -> (batch_size, n_oprs, opr_feat_dim)
            batch_station_features: 工位特征 shape -> (batch_size, n_stations, station_feat_dim)
        """
        def build_opr_features():
            """构建工序特征信息
            Features:
                status: 工序状态, 1: 已排产 0: 未排产
                n_neighbor_stations: 邻近的工位数(即工序可分配的工位数)
                process_time: 平均加工时长
                n_unscheduled_oprs: 订单中未排产的工序数
                completion_time: 任务(订单)完工时间
                start_time: 工序开始时间
            """
            n_features = 6
            opr_features = torch.zeros(n_max_oprs, n_features)

            # 统计不同任务的工序个数
            job_opr_map = collections.defaultdict(set)
            for opr in oprs.values():
                job_opr_map[opr.job_index].add(opr.index)
            
            for opr_key, stations in opr_stations.items():
                opr = oprs[opr_key]
                opr_idx = opr.index
                # 获取工序可分配的工位数
                opr_features[opr_idx, 1] = len(stations)
                # 获取工序在不同工位上的平均加工时长
                process_time = list(opr.process_time.values())
                opr_features[opr_idx, 2] = sum(process_time) // len(process_time)
                # 获取订单中未排产的工序数
                opr_features[opr_idx, 3] = len(job_opr_map[opr.job_index])
                
            return opr_features

        def build_station_features():
            """构建工位特征信息
            Features:
                n_neighbor_oprs: 邻近的工序数(即工位可处理的工序数)
                available_time: 工位可开始时间, 从0开始
                utilization: 当前工位处理工序的进度, 取值[0, 1]
            """
            n_features = 3
            station_features = torch.zeros(n_max_stations, n_features)
            for station_key, oprs in station_oprs.items():
                station_idx = stations[station_key].index
                # 获取工位可处理工序数
                station_features[station_idx, 0] = len(oprs)

            return station_features

        def build_om_edge_features():
            """构建Operation-Machine边的特征
            """
            edge_features = torch.zeros((n_max_oprs, n_max_stations))
            for opr_key, link_stations in opr_stations.items():
                for link_station in link_stations:
                    edge_features[oprs[opr_key].index, link_station.index] = \
                            oprs[opr_key].process_time[link_station.station_key]
            return edge_features

        def obtain_opr_station():
            """获取工序工位邻接矩阵
            """
            opr_station_indices = {opr_key : [station.index for station in alloc_stations] 
                                for opr_key, alloc_stations in opr_stations.items()}
            opr_station = torch.zeros((n_max_oprs, n_max_stations))
            for opr_key, link_stations in opr_station_indices.items():
                opr_station[oprs[opr_key].index, link_stations] = 1
            return opr_station

        def obtain_num_opr_and_opr_job_and_first_opr():
            """获取不同任务(订单)的工序个数及工序对应任务及首工序索引
            """
            job_opr_map = collections.defaultdict(set)
            for opr in oprs.values():
                job_opr_map[opr.job_index].add(opr.index)
            n_jobs = len(job_opr_map)
            num_opr = torch.zeros((n_max_job,))
            opr_job = torch.zeros((n_max_oprs,))
            first_opr = torch.zeros((n_max_job,))
            s = 0
            for i in range(n_jobs):
                e = len(job_opr_map[i])
                e += s
                num_opr[i] = e - s
                opr_job[s:e] = i
                first_opr[i] = s
                s = e
            return num_opr, opr_job, first_opr
        
        def obtain_opr_next():
            """获取工序的后继工序
            (batch_size, n_oprs, n_oprs)
            """
            opr_next = torch.zeros((n_max_oprs, n_max_oprs))
            for opr_key, links in opr_links.items():
                for opr in links:
                    opr_next[oprs[opr_key].index, opr.index] = 1
            return opr_next

        def obtain_opr_pre(opr_next):
            """获取工序的前向工序
            """
            return opr_next.t()

        def obtain_line_opr_pre():
            """获取每条线上的工序的前向工序
            """
            pass
        
        def obtain_line_info():
            """获取主线与部装线相关信息
            Return
            --------
                mainline_flag: 是否是主线的标志
                            0: 非主线, 1: 主线
                line_links: 主线与部装线的工序链接标志
                            False: 非主线-部装线链接, True: 主线-部装线链接
                subline_links: 所有部装线工序链接的主线工序标志
            """
            mainline_flag = torch.zeros((n_max_oprs,))
            line_links = torch.full((n_max_oprs, n_max_oprs), fill_value=False, dtype=torch.bool)
            subline_links = torch.full((n_max_oprs, n_max_oprs), fill_value=False, dtype=torch.bool)
            for opr in oprs.values():
                if opr.is_mainline:
                    mainline_flag[opr.index] = 1
                    continue
                for next_opr in opr_links[opr.opr_key]:
                    if next_opr.is_mainline:
                        line_links[next_opr.index, opr.index] = True
                sub = []
                tmp_opr = copy.deepcopy(opr)
                while not tmp_opr.is_mainline:
                    sub.append(tmp_opr.index)
                    #NOTE 目前工序只能有一个后继工序
                    tmp_opr = opr_links[tmp_opr.opr_key][0]
                subline_links[sub, tmp_opr.index] = True
            return mainline_flag, line_links, subline_links
        
        def obtain_mainline_num_opr():
            """获取各个任务(订单)中各个主线的工序数
            """
            job_opr_map = collections.defaultdict(set)
            for opr in oprs.values():
                if opr.is_mainline:
                    job_opr_map[opr.job_index].add(opr.index)
            n_jobs = len(job_opr_map)
            mainline_num_opr = torch.zeros((n_max_job,))
            for i in range(n_jobs):
                mainline_num_opr[i] = len(job_opr_map[i])
            return mainline_num_opr

        def obtain_opr_line_index():
            """获取各个工序所在线的索引
            """
            opr_line_index = torch.zeros(size=(n_max_oprs,))
            for opr in oprs.values():
                opr_line_index[opr.index] = opr.line_index
            return opr_line_index


        batch_opr_features, batch_station_features, batch_edge_features = [], [], []
        batch_opr_station, batch_num_opr, batch_opr_job, batch_first_opr = [], [], [], []
        batch_opr_next, batch_opr_pre = [], []
        batch_mainline_flag, batch_line_links, batch_subline_links = [], [], []
        batch_mainline_num_opr = []
        batch_opr_line_index = []
        batch_jobs = []
        # 获取该批次最大工序数
        n_max_oprs = max(list(map(lambda x : len(x), batch_oprs)))
        n_max_stations = 0
        for each_item in batch_opr_stations:
            tmp = set()
            for sts in each_item.values():
                for st in sts:
                    tmp.add(st.index)
            tmp.add(n_max_stations)
            n_max_stations = max(tmp)
        n_max_stations += 1
        job_opr_map = collections.defaultdict(set)
        n_max_job = 0
        for oprs in batch_oprs:
            for opr in oprs.values():
                job_opr_map[opr.job_index].add(opr.index)
            n_max_job = max(n_max_job, len(job_opr_map))
        batch_size = len(batch_oprs)
        for i in range(batch_size):
            oprs, opr_stations, opr_links = batch_oprs[i], batch_opr_stations[i], batch_opr_links[i]
            stations = {item.station_key: item for alloc_stations in list(opr_stations.values()) for item in alloc_stations}
            # 记录工位可处理工序列表
            station_oprs = collections.defaultdict(list)
            for opr_key, alloc_stations in opr_stations.items():
                for station in alloc_stations:
                    station_oprs[station.station_key].append(oprs[opr_key])

            batch_opr_features.append(build_opr_features())
            batch_station_features.append(build_station_features())
            batch_edge_features.append(build_om_edge_features())
            batch_opr_station.append(obtain_opr_station())
            num_opr, opr_job, first_opr = obtain_num_opr_and_opr_job_and_first_opr()
            batch_num_opr.append(num_opr)
            batch_opr_job.append(opr_job)
            batch_first_opr.append(first_opr)
            batch_opr_next.append(obtain_opr_next())
            batch_opr_pre.append(obtain_opr_pre(batch_opr_next[-1]))
            mainline_flag, line_links, subline_links = obtain_line_info()
            batch_mainline_flag.append(mainline_flag)
            batch_line_links.append(line_links)
            batch_subline_links.append(subline_links)
            batch_mainline_num_opr.append(obtain_mainline_num_opr())
            batch_opr_line_index.append(obtain_opr_line_index())
        source_data = {}
        batch_opr_features = torch.stack(batch_opr_features, dim=0)
        batch_station_features = torch.stack(batch_station_features, dim=0)
        batch_edge_features = torch.stack(batch_edge_features, dim=0)
        batch_opr_station = torch.stack(batch_opr_station, dim=0).long()
        batch_num_opr = torch.stack(batch_num_opr, dim=0).int()
        batch_opr_job = torch.stack(batch_opr_job, dim=0).long()
        batch_first_opr = torch.stack(batch_first_opr, dim=0).long()
        batch_jobs = torch.tensor(batch_jobs, dtype=torch.long)
        batch_opr_next = torch.stack(batch_opr_next, dim=0).int()
        batch_opr_pre = torch.stack(batch_opr_pre, dim=0).int()
        batch_mainline_flag = torch.stack(batch_mainline_flag, dim=0).int()
        batch_line_links = torch.stack(batch_line_links, dim=0)
        batch_subline_links = torch.stack(batch_subline_links, dim=0)
        batch_mainline_num_opr = torch.stack(batch_mainline_num_opr, dim=0).int()
        batch_opr_line_index = torch.stack(batch_opr_line_index, dim=0).int()
        source_data['opr_features'] = batch_opr_features
        source_data['station_features'] = batch_station_features
        source_data['edge_features'] = batch_edge_features
        source_data['opr_station'] = batch_opr_station
        source_data['num_opr'] = batch_num_opr
        source_data['opr_job'] = batch_opr_job
        source_data['first_opr'] = batch_first_opr
        source_data['opr_next'] = batch_opr_next
        source_data['opr_pre'] = batch_opr_pre
        source_data['mainline_flag'] = batch_mainline_flag
        source_data['line_links'] = batch_line_links
        source_data['subline_links'] = batch_subline_links
        source_data['mainline_num_opr'] = batch_mainline_num_opr
        source_data['opr_line_index'] = batch_opr_line_index
        source_data['jobs'] = batch_jobs
        return source_data
        
    @staticmethod
    def init_state(batch_oprs: list, batch_opr_stations: list, batch_opr_links: list, **params):
        """初始化状态
        """
        source_data = StateUtils.build_features(batch_oprs, batch_opr_stations, batch_opr_links, **params)
        state = State(source_data['opr_features'], source_data['station_features'], source_data['edge_features'],
                        source_data['opr_station'], source_data['num_opr'], 
                        source_data['opr_job'], source_data['first_opr'],
                        source_data['opr_next'], source_data['opr_pre'], 
                        source_data['mainline_flag'], source_data['line_links'], source_data['subline_links'],
                        source_data['mainline_num_opr'], source_data['opr_line_index'],
                        source_data['jobs'], params)
        # 工序特征: 工序开始时间
        StateUtils.init_opr_start_time(state)
        # 工序特征: 任务完工时间
        StateUtils.estimate_job_completion_time(state, state.batch_indices)
        # 更新工序开始时间及任务完工时间(当有部装线存在时)
        StateUtils.update_opr_start_time(state, state.batch_indices)

        
        return state

    @staticmethod
    def state_transfer(state: State, action: Action, **params):
        """状态转移
        """
        def to_next_time(batch_indices, oprs):
            """移动到下一时刻
            """

            def exist_eligible_action():
                """当前时刻(batch_time)是否存在合法动作(工序到工位的映射)
                """
                # 得到当前状态下的候选动作集, shape -> (batch_size, n_oprs, n_stations)
                eligible_actions = StateUtils.obtain_ready_actions(next_state, memory)
                # 各批次是否存在合法动作(True: 合法, False: 非法), shape -> (batch_size,)
                eligible = eligible_actions.any(dim=-1).any(dim=-1)
                return eligible

            def update_mask_now():
                complete_jobs = torch.where(complete_stations, next_state.batch_station_schedule[batch_indices, :, 3].double(), -1.0).float()
                jobs_index = np.argwhere(complete_jobs.cpu() >= 0).to(next_state.device)
                job_idxes = complete_jobs[jobs_index[0], jobs_index[1]].long()
                batch_idxes = jobs_index[0]

                select_part = next_state.batch_mask_busy_station[batch_indices]
                select_part[complete_stations] = False
                next_state.batch_mask_busy_station[batch_indices] = select_part
                next_state.batch_mask_busy_job[batch_idxes, job_idxes] = False

            # 如果任务都已完成,则不需要进行时间步的转移
            if next_state.batch_done.all():
                return
            
            # 若当前时刻不存在合法动作且还存在未排产的工序,则需要移动到下一时刻, shape -> (batch_size,)
            is_transit = ~exist_eligible_action()
            if not next_state.is_train:
                is_transit = is_transit & ~next_state.batch_done
            is_transit = is_transit[batch_indices]
            while is_transit.any():
                # 获取不同工位可开始的时间, shape -> (batch_size, n_stations)
                available_time = next_state.batch_station_schedule[batch_indices, : , 1]
                # 获取未排工序最小可开始时间
                cond1 = (next_state.batch_opr_features[batch_indices, :, 0] == 0).squeeze(-1)
                cond2 = (next_state.batch_opr_features[batch_indices, :, 5] > 0).squeeze(-1)
                cond = cond1 & cond2 & next_state.batch_valid_oprs[batch_indices]
                # shape -> (batch_size, n_oprs)
                max_time = torch.max(next_state.batch_opr_features[batch_indices, :, 5].squeeze(-1), dim=1).values.unsqueeze(-1).expand_as(cond)
                start_time = torch.where(cond, next_state.batch_opr_features[batch_indices, :, 5].squeeze(-1), max_time)
                min_opr_start_time = torch.min(start_time, dim=1).values
                max_station_time = torch.max(available_time, dim=1).values
                sel_t = torch.where(min_opr_start_time > max_station_time, min_opr_start_time, max_station_time)
                # 保留不同工位可开始时间的最小值作为下一个时间步
                max_available_time = torch.where(available_time > next_state.batch_time[batch_indices, None], 
                        available_time, 
                        sel_t.unsqueeze(-1).expand_as(available_time))
                next_time = torch.min(max_available_time, dim=1)[0]
                next_state.batch_time[batch_indices] = torch.where(is_transit, next_time, 
                                                                   next_state.batch_time[batch_indices])

                # Detect the machines that completed (at above time)
                complete_stations = torch.where((available_time == next_time[:, None]) & 
                            (next_state.batch_station_schedule[batch_indices, :, 0] == 0) & is_transit[:, None], True, False)
                # Update partial schedule (state), variables and feature vectors
                aa = next_state.batch_station_schedule.transpose(1, 2)[batch_indices]
                aa[complete_stations, 0] = 1
                next_state.batch_station_schedule[batch_indices] = aa.transpose(1, 2)

                utiliz = next_state.batch_station_schedule[:, :, 2]
                cur_time = next_state.batch_time[:, None].expand_as(utiliz)
                utiliz = torch.minimum(utiliz, cur_time)
                utiliz = utiliz.div(next_state.batch_time[:, None] + 1e-5)
                next_state.batch_station_features[:, :, 2] = utiliz

                update_mask_now()
                is_transit = ~exist_eligible_action()
                if not next_state.is_train:
                    is_transit = is_transit & ~next_state.batch_done
                is_transit = is_transit[batch_indices]

        def filter_no_action(oprs, stations, jobs):
            """过滤当前时刻不采取动作的实例
            """
            n_oprs = next_state.n_total_opr[next_state.batch_indices]
            # 如果当前执行的是首工序,则打开`NO_ACT`,否则关闭
            is_first = next_state.is_first[next_state.batch_indices, oprs]
            mask_first = torch.where((oprs < n_oprs-1) & is_first, True, False)
            next_state.open_no_act[next_state.batch_indices[mask_first], stations[mask_first]] = True
            # 不采用动作的实例掩码
            mask_instances = torch.where(oprs < n_oprs-1, True, False)
            _oprs = oprs[mask_instances]
            _stations = stations[mask_instances]
            _jobs = jobs[mask_instances]
            _batch_indices = next_state.batch_indices[mask_instances]
            # 针对不采取动作的实例,执行特殊动作`NO_ACT`
            if (~mask_instances).any():
                # 当前工位暂时不能执行特殊动作`NO_ACT`
                next_state.open_no_act[next_state.batch_indices[~mask_instances], stations[~mask_instances]] = False
                select_indices = next_state.batch_indices[~mask_instances]
                select_oprs = oprs[~mask_instances]
                select_stations = stations[~mask_instances]
                select_jobs = jobs[~mask_instances]
                select_proc_times = next_state.batch_opr_proctime[select_indices, select_oprs, select_stations] \
                * next_state.steps[select_indices]
                
                # Update feature vectors of operations
                select_times = next_state.batch_opr_features[select_indices, select_oprs, 5]
                next_state.batch_station_schedule[select_indices, select_stations, 1] = next_state.batch_time[select_indices] + select_proc_times
                next_state.batch_station_schedule[select_indices, select_stations, 2] += select_proc_times
                # 更新工位已分配任务
                next_state.batch_station_schedule[select_indices, select_stations, 3] = select_jobs.float()
                # 更新工序任务索引
                next_state.batch_opr_schedule[select_indices, select_oprs, 4] = select_jobs.float()
                # Update feature vectors of machines
                next_state.batch_station_features[select_indices, select_stations, 1] = next_state.batch_time[select_indices] + select_proc_times
                utiliz = next_state.batch_station_schedule[select_indices, :, 2]
                cur_time = next_state.batch_time[select_indices, None].expand_as(utiliz)
                utiliz = torch.minimum(utiliz, cur_time)
                utiliz = utiliz.div(next_state.batch_time[select_indices, None] + 1e-9)
                next_state.batch_station_features[select_indices, :, 2] = utiliz
                # 更新工位状态
                occupied = torch.where(select_proc_times > 0, True, False)
                next_state.batch_mask_busy_station[select_indices[occupied], select_stations[occupied]] = True
                next_state.batch_station_schedule[select_indices[occupied], select_stations[occupied], 0] = 0
                
                # 更新`NO_ACT`工序所在工位可处理的所有未排工序的开始时间及对应任务的结束时间
                # shape -> (batch_size, n_oprs)
                cond1 = next_state.batch_opr_station[select_indices, :, select_stations] == 1
                # shape -> (batch_size, n_oprs)
                cond2 = next_state.batch_opr_features[select_indices, :, 0] == 0
                cond = cond1 & cond2
                # shape -> (batch_size, n_oprs)
                start_times = next_state.batch_opr_features[select_indices, :, 5].squeeze(-1)
                # shape -> (batch_size,1)
                avai_times = next_state.batch_station_features[select_indices, select_stations, 1].unsqueeze(-1)
                #print(avai_times.shape)
                avai_times = avai_times.expand_as(start_times)
                #print(avai_times.shape)
                avai_times = torch.where(cond, avai_times, -avai_times)
                #print(avai_times)
                max_start_times = torch.where(avai_times > start_times, avai_times, start_times)
                #print(max_start_times)
                batch_ori_start_time = next_state.batch_opr_features[select_indices, :, 5]. \
                    gather(1, next_state.batch_first_opr[select_indices, ...])
                #next_state.batch_opr_features[select_indices, :, 5] = max_start_times
                batch_start_time = max_start_times. \
                    gather(1, next_state.batch_first_opr[select_indices, ...])
                time_steps = batch_start_time - batch_ori_start_time
                add_time = time_steps.gather(1, next_state.batch_opr_job[select_indices])
                next_state.batch_opr_features[select_indices, :, 5] += add_time
                next_state.batch_opr_features[select_indices, select_oprs, 5] = select_times
                #print(next_state.batch_opr_features[select_indices[0], :, :])
                StateUtils.estimate_job_completion_time(next_state, select_indices)
                to_next_time(select_indices, select_oprs)
            
            return _oprs, _stations, _jobs, _batch_indices

        def remove_unselected_pair():
            """删除与排产工序相关的工序-工位连接(保留排产工序的工序-工位对)
            """
            remain_opr_station = torch.zeros(size=(next_state.batch_size, next_state.n_stations), dtype=torch.int64)
            remain_opr_station[batch_indices, stations] = 1
            next_state.batch_opr_station[batch_indices, oprs] = remain_opr_station[batch_indices, :]
            next_state.batch_opr_proctime *= next_state.batch_opr_station
            next_state.batch_edge_features = copy.deepcopy(next_state.batch_opr_proctime)
            next_state.batch_opr_station_remain &= next_state.batch_opr_station
            next_state.batch_opr_station_remain[batch_indices, oprs, stations] = 0

        def update_opr_features():
            """更新工序特征
            """
            # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
            next_state.batch_opr_features[batch_indices, oprs, :3] = \
                        torch.stack((torch.ones(batch_indices.size(0), dtype=torch.float),
                                    torch.ones(batch_indices.size(0), dtype=torch.float),
                                    proc_times), dim=1)

            # Update 'Number of unscheduled operations in the job'
            start_opr = next_state.batch_first_opr[batch_indices, jobs]
            end_opr = next_state.batch_last_opr[batch_indices, jobs]
            for i in range(batch_indices.size(0)):
                next_state.batch_opr_features[batch_indices[i], start_opr[i]:end_opr[i]+1, 3] -= 1

            # Update 'Start time' and 'Job completion time'
            StateUtils.estimate_opr_start_time(next_state, oprs, jobs, batch_indices)
            StateUtils.estimate_job_completion_time(next_state, batch_indices)
            # 更新工序开始时间及任务完工时间(当有部装线存在时)
            StateUtils.update_opr_start_time(next_state, batch_indices, oprs)

            next_state.batch_makespan = torch.max(next_state.batch_opr_features[:, :, 4], dim=1)[0]
            

        def update_schedule_state():
            """更新调度状态信息
            """
            # 更新工序状态及分配的工位
            next_state.batch_opr_schedule[batch_indices, oprs, :2] = torch.stack((torch.ones(batch_indices.size(0)), stations), dim=1)
            # 更新工序开始时间
            next_state.batch_opr_schedule[batch_indices, oprs, 2] = next_state.batch_opr_features[batch_indices, oprs, 5]
            # 更新工序完工时间
            next_state.batch_opr_schedule[batch_indices, oprs, 3] = next_state.batch_opr_features[batch_indices, oprs, 5] + \
                                                                                next_state.batch_opr_features[batch_indices, oprs, 2]
            # 更新工序任务索引
            next_state.batch_opr_schedule[batch_indices, oprs, 4] = next_state.batch_opr_job[batch_indices, oprs].float()

            # 更新工位状态: 因为此时正在处理工序,所以更新状态为忙碌
            next_state.batch_station_schedule[batch_indices, stations, 0] = torch.zeros(batch_indices.size(0))
            # 更新工位的可开始时间
            next_state.batch_station_schedule[batch_indices, stations, 1] = next_state.batch_time[batch_indices] + proc_times
            # 更新工位产能占用时间
            next_state.batch_station_schedule[batch_indices, stations, 2] += proc_times
            # 更新工位已分配任务
            next_state.batch_station_schedule[batch_indices, stations, 3] = jobs.float()

        def update_station_features():
            """更新工位(机器)信息
            """
            next_state.batch_station_features[batch_indices, :, 0] = \
                    torch.count_nonzero(next_state.batch_opr_station[batch_indices, ...], dim=1).float()
            next_state.batch_station_features[batch_indices, stations, 1] = next_state.batch_time[batch_indices] + proc_times
            utiliz = next_state.batch_station_schedule[batch_indices, :, 2]
            cur_time = next_state.batch_time[batch_indices, None].expand_as(utiliz)
            utiliz = torch.minimum(utiliz, cur_time)
            utiliz = utiliz.div(next_state.batch_time[batch_indices, None] + 1e-9)
            next_state.batch_station_features[batch_indices, :, 2] = utiliz

        def update_mask():
            """更新掩码
            """
            next_state.batch_mask_finish_opr[batch_indices, oprs] = True
            next_state.batch_mask_finish_job = torch.where(next_state.n_scheduled_job_opr==next_state.batch_num_opr,
                                                    True, next_state.batch_mask_finish_job)
            # 针对虚拟工序节点,不占用工位产能
            occupied = torch.where(proc_times > 0, True, False)
            next_state.batch_mask_busy_station[batch_indices[occupied], stations[occupied]] = True
            next_state.batch_mask_busy_job[batch_indices, jobs] = True

        def update_unfinish_instance():
            """更新未完成的实例
            """
            if next_state.exist_no_action:
                remain_station_oprs = []
                if not next_state.is_train:
                    no_act_site = next_state.n_total_opr - 1
                    for b in range(next_state.batch_size):
                        remain_station_oprs.append(
                            torch.count_nonzero(next_state.batch_opr_station_remain[b, :no_act_site[b], :], dim=0))
                    remain_station_oprs = torch.stack(remain_station_oprs, dim=0)
                else:
                    # 各个工位剩余可处理的工序数(排除掉`NO_ACT`), shape -> (batch_size, n_stations)
                    remain_station_oprs = torch.count_nonzero(next_state.batch_opr_station_remain[:, :-1, :], dim=1)
                # 获取没有可处理的工序的工位, shape -> (batch_size, n_stations)
                mask = torch.where(remain_station_oprs <= 0, True, False)
                next_state.batch_done = mask.all(dim=1)
                #print(next_state.batch_done)
                # 如果某批次已经完成,为了保证训练阶段各批次完成进度一致,完成的批次可以执行`NO_ACT`
                dones = torch.where(next_state.batch_done)[0]
                if next_state.is_train and dones.size(0) > 0:
                    mask[dones, 0] = False
                mask_finish = mask.all(dim=1)
                if mask_finish.any():
                    next_state.batch_indices = torch.arange(next_state.batch_size)[~mask_finish]
            else:
                next_state.batch_done = next_state.batch_mask_finish_job.all(dim=1)
                mask_finish = next_state.n_scheduled_opr < next_state.n_total_opr
                if ~(mask_finish.all()):
                    next_state.batch_indices = torch.arange(next_state.batch_size)[mask_finish]

        oprs = action.opr_station_pair[0, :]
        stations = action.opr_station_pair[1, :]
        jobs = action.opr_station_pair[2, :]
        memory = params['memory']
        next_state = copy.deepcopy(state)
        
        # 是否存在动作: '不采取动作'
        if next_state.exist_no_action:
            oprs, stations, jobs, batch_indices = filter_no_action(oprs, stations, jobs)
        else:
            batch_indices = next_state.batch_indices
        
        if oprs.size(0) == 0:
            return next_state

        # 每个实例已完成的工序数 (batch_size,)
        next_state.n_scheduled_opr[batch_indices] += 1
        # 每个任务已完成的工序数 (batch_sie, n_jobs)
        next_state.n_scheduled_job_opr[batch_indices, jobs] += 1
        # 每个工序的依赖工序 (batch_size, n_oprs, n_oprs)
        next_state.batch_opr_depencies[batch_indices, :, oprs] = 0

        # Removed unselected O-M arcs of the scheduled operations
        remove_unselected_pair()

        proc_times = next_state.batch_opr_proctime[batch_indices, oprs, stations]

        # Update feature vectors of operations
        update_opr_features()

        # Update partial schedule (state)
        update_schedule_state()

        # Update feature vectors of machines
        update_station_features()

        # Update other variable according to actions
        update_mask()

        # Update the vector for uncompleted instances
        update_unfinish_instance()

        # 移动时间窗口到下一时间步(即移动batch_time)
        to_next_time(batch_indices, oprs)

        return next_state
    
    def update_opr_start_time(state: State, batch_indices: torch.Tensor, oprs: torch.Tensor = None):
        """当存在部装线时,需要考虑部装线的完工时间,从而更新与其链接的主线工序开始时间
        """
        #print('=======================================')
        # 不含部装线
        if not state.exist_subline:
            return
        for idx, b in enumerate(batch_indices):
            # 遍历该实例下的所有任务
            s = 0
            _s = 0
            n_total_oprs = 0
            for j, c in enumerate(state.batch_num_opr[b, ...]):
                c = c.item()
                #print(j, c)
                #print(torch.where(state.batch_line_links[b, s:s+c, s:s+c]))
                # 获取有部装线链接的主线工序 shape -> (n_oprs,)
                mainline_oprs = torch.where(state.batch_line_links[b, s:s+c, s:s+c])[0] + s
                if mainline_oprs.size(0) == 0:
                    continue
                #print(mainline_oprs)
                # 获取主线工序链接的部装线工序
                subline_oprs = torch.where(state.batch_line_links[b, s:s+c, s:s+c])[1] + s
                #print(subline_oprs)
                # 获取链接的部装线工序完工时间
                subline_opr_end_time = state.batch_opr_features[b, subline_oprs, 5] + \
                                            state.batch_opr_features[b, subline_oprs, 2]
                #print(subline_opr_end_time)
                # 获取主线工序的开始时间
                mainline_opr_start_time = state.batch_opr_features[b, mainline_oprs, 5]
                #print(mainline_opr_start_time)
                # 获取主线工序开始时间与所链接部装工序的完工时间的最大差值
                max_step = torch.max(subline_opr_end_time - mainline_opr_start_time)
                #print(max_step)
                # 更新主线工序开始时间(主线工序整体右移)
                mainline_opr_indices = torch.where(state.batch_mainline_flag[b])[0]
                #print(mainline_opr_indices)
                # 获取当前任务的主线工序个数
                m_num = state.batch_mainline_num_opr[b, j].item()
                #print(m_num)
                _e = _s + m_num
                # 选择待更新的主线工序
                cur_opr = None
                if oprs is not None:
                    cur_opr = oprs[idx].squeeze()
                    # 判断该工序是否落在当前任务上
                    if cur_opr < s or cur_opr >= s+c:
                        cur_opr = None
                # 当前处理的工序为部装线工序
                #print(cur_opr)
                if cur_opr and not state.batch_mainline_flag[b, cur_opr]:
                    #pass
                    # 记录主线列表的当前位置
                    _tmp_s = _s
                    # 找到与该部装线链接的主线工序
                    _s = torch.where(state.batch_subline_links[b, cur_opr, :])[0]
                    # 假设共两个任务,每个任务的工序数为56个,主线工序数为20个,则第56道工序对应的应该是第二个任务的第一道主线工序
                    # 对应主线列表中的第20道工序
                    _s =  _tmp_s + (_s - n_total_oprs)
                if max_step > 0:
                    #print(mainline_opr_indices.shape)
                    #print(_s, _e)
                    #print(mainline_opr_indices[_s:_e])
                    # 获取未排产的主线工序
                    schedule_mask = torch.where(state.batch_opr_features[b, mainline_opr_indices[_s:_e], 0] == 0, 
                                                  True, False)
                    #print(schedule_mask)
                    state.batch_opr_features[b, mainline_opr_indices[_s:_e][schedule_mask], 5] += max_step
                #print(mainline_opr_indices[_s:s+c])
                n_total_oprs += c
                s += c
                _s = _e
        StateUtils.estimate_job_completion_time(state, batch_indices)

    @staticmethod
    def init_opr_start_time(state: State):
        """工序开始时间初始化预估
        考虑部装线
        """
        state.batch_opr_features[..., 5] = torch.bmm(state.cumsum_opr, 
                    state.batch_opr_features[...,  2].unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def estimate_job_completion_time(state: State, batch_indices: torch.Tensor):
        """任务完工时间预估
        考虑部装线
        """
        batch_end_opr = state.batch_first_opr[batch_indices, ...] + state.batch_mainline_num_opr[batch_indices, ...] - 1
        batch_end_opr = torch.where(batch_end_opr >= 0, batch_end_opr, 0)
        #print(batch_end_opr[0])
        #print(state.batch_opr_features[batch_indices[0], :, 5])
        # 得到每个任务的完工时间, shape -> (batch_size, n_jobs)
        batch_end_time = (state.batch_opr_features[batch_indices, :, 5] + \
                    state.batch_opr_features[batch_indices, :, 2]).gather(1, batch_end_opr)
        #print(batch_end_time[0])
        state.batch_opr_features[batch_indices, :, 4] = batch_end_time.gather(1, state.batch_opr_job[batch_indices, ...])

    @staticmethod
    def estimate_opr_start_time(state: State, oprs: torch.Tensor, jobs: torch.Tensor, batch_indices: torch.Tensor):
        """工序开始时间预估
        考虑部装线
        """
        # 得到前一个工序
        last_oprs = torch.where(oprs - 1 < state.batch_first_opr[batch_indices, jobs], 
                                state.n_total_opr[batch_indices] - 1, oprs - 1)
        state.cumsum_opr[batch_indices, :, last_oprs] = 0
        state.batch_opr_features[batch_indices, oprs, 5] = state.batch_time[batch_indices]
        is_scheduled = state.batch_opr_features[batch_indices, :, 0]
        # 只考虑当前工序所在的线(主线或部装线),其他线的工序全部Mask掉, shape -> (batch_size,)
        line_index = state.batch_opr_line_index[batch_indices, oprs].unsqueeze(-1)
        is_mask = torch.where(state.batch_opr_line_index[batch_indices] == line_index, 0, 1)
        # 记录非当前线的其他工序开始时间
        origin_start_time = state.batch_opr_features[batch_indices, :, 5] * is_mask
        mean_proc_time = state.batch_opr_features[batch_indices, :, 2]
        # real start time of scheduled operations
        start_times = state.batch_opr_features[batch_indices, :, 5] * is_scheduled
        un_scheduled = 1 - is_scheduled  # unscheduled operations
        # estimate start time of unscheduled operations
        estimate_times = torch.bmm(state.cumsum_opr[batch_indices, ...].double(), 
                    (start_times+mean_proc_time).unsqueeze(-1).double()).squeeze(-1) * un_scheduled
        estimate_start_time = start_times.float() + estimate_times.float()
        state.batch_opr_features[batch_indices, :, 5] = origin_start_time + estimate_start_time * (1-is_mask)

    @staticmethod
    def obtain_om_pairs(batch_oprs: list, batch_opr_stations: list):
        """获取工序工位对集合 O-M pair
        """
        om_pairs = [collections.defaultdict(list) for _ in range(len(batch_oprs))]
        for i in range(len(batch_oprs)):
            oprs, opr_stations = batch_oprs[i], batch_opr_stations[i]
            for opr_key, links in opr_stations.items():
                opr = oprs[opr_key]
                for link in links:
                    om_pairs[i][opr.index].append((opr.index, link.index, opr.job_index))
        return om_pairs

    @staticmethod
    def obtain_ready_actions(state: State, memory: Memory):
        """获取当前状态下的候选动作集
        """
        batch_size, n_oprs, n_stations = state.batch_opr_station.size()
        n_jobs = state.n_jobs

        eligible = copy.deepcopy(state.batch_ori_opr_station)
        
        # 工序的依赖工序 (batch_size, n_oprs, n_oprs)
        batch_opr_depencies = state.batch_opr_depencies
        # 工序未处理的依赖工序个数 (batch_size, n_oprs)
        n_depencies = batch_opr_depencies.sum(dim=-1).squeeze(-1)
        # 过滤掉已排产的工序
        n_depencies[state.batch_mask_finish_opr] = -1
        # 当前就绪工序
        no_depencies = torch.where(n_depencies==0, True, False).unsqueeze(-1).expand_as(eligible)

        eligible[:] = eligible * no_depencies
        
        if state.exist_no_action:
            # 针对训练完成的批次可以执行NO_ACT
            padding = torch.full(size=(batch_size, n_oprs, n_stations), dtype=torch.bool, fill_value=False)
            padding[:, :, 0] = True
            dones = torch.where(state.batch_done, True, False) # (batch_size,)
            padding[:] = padding * (dones.unsqueeze(-1).unsqueeze(-1).expand_as(eligible) & state.is_train)
            # 打开或关闭NO_ACT
            # (batch_size, n_oprs, n_stations)
            no_act_job = torch.where(state.batch_opr_job == n_jobs-1, True, False).unsqueeze(-1).expand_as(eligible) 
            open_no_act = state.open_no_act.unsqueeze(-2).expand_as(eligible)
            opened = torch.where(open_no_act, open_no_act*no_act_job, ~no_act_job)

            eligible[:] = (eligible & (padding | opened))
        
        eligible = eligible.bool()
        eligible = RuleFilter.choose(eligible, state, memory)
        return eligible
