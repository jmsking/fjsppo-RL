import numpy as np
from typing import Union, List
from functools import cmp_to_key
import collections
import copy
import os
import json
import random
from datetime import datetime
from project.domain.operation import Operation
from project.domain.calendar import Calendar
from project.domain.station import Station
from project.domain.period import Period
from project.utils.json_utils import JsonUtils
from project.utils.load_utils import LoadUtils
from project.common.constant import DIR



class CaseUtils:
    """样例生成器
    """

    @staticmethod
    def generate(source='random', is_same_opr=True, no_action=False, **kwargs):
        batch_size = kwargs.get('batch_size', 0)
        n_jobs = kwargs.get('n_jobs', 0)
        n_stations = kwargs.get('n_stations', 0)
        is_save = kwargs.get('is_save', False)
        exist_subline = kwargs.get('exist_subline', False)
        is_single_station = kwargs.get('is_single_station', False)
        file_path = kwargs.get('file_path', '')
        batch_oprs, batch_opr_stations, batch_opr_links = [], [], []
        # 主线或者部装线数量
        n_lines = []
        # 随机产生样本
        if source == 'random':
            ratio = 1
            oprs_per_job_min = int(n_stations * 0.8)
            oprs_per_job_max = int(n_stations * ratio)
            if is_single_station:
                oprs_per_job_min = oprs_per_job_max
            oprs_per_job = [np.random.randint(oprs_per_job_min, oprs_per_job_max+1) for _ in range(n_jobs)]
            for i in range(batch_size):
                if not is_same_opr:
                    oprs_per_job = [np.random.randint(oprs_per_job_min, oprs_per_job_max+1) for _ in range(n_jobs)]
                oprs, opr_stations, opr_links, line_index = CaseUtils.random_produce(i, n_jobs, oprs_per_job, n_stations, 
                                                                                     exist_subline=exist_subline,
                                                                                     is_single_station=is_single_station)
                batch_oprs.append(oprs)
                batch_opr_stations.append(opr_stations)
                batch_opr_links.append(opr_links)
                n_lines.append(line_index)
        # 随机样本文件
        elif source == 'file':
            file_path = kwargs['file_path']
            print(f'start loading file from path: `{file_path}`')
            data_files = os.listdir(file_path)
            for i in range(len(data_files)):
                f = file_path+data_files[i]
                oprs, opr_stations, opr_links, line_index = CaseUtils.load_from_file(f)
                if line_index < 0:
                    continue
                batch_oprs.append(oprs)
                batch_opr_stations.append(opr_stations)
                batch_opr_links.append(opr_links)
                n_lines.append(line_index)
        # 真实排产场景数据
        elif source == 'as':
            file_path = kwargs['file_path']
            print(f'start loading file from path: `{file_path}`')
            for b in range(batch_size):
                oprs, opr_stations, opr_links, line_index = CaseUtils.produce_from_file(file_path)
                batch_oprs.append(oprs)
                batch_opr_stations.append(opr_stations)
                batch_opr_links.append(opr_links)
                n_lines.append(line_index)
        elif source == 'simulation':
            print(f'simulation -> {file_path}')
            batch_oprs, batch_opr_stations, batch_opr_links = LoadUtils.load_simulation(file_path)
        else:
            raise ValueError('Case generator only support `random` or `file` or `as` or `simulation`')
        if no_action:
            for b in range(batch_size):
                n_max_lines = n_lines[b]
                n_max_oprs = len(batch_oprs[b])
                stations = []
                for opr_key, alloc_stations in batch_opr_stations[b].items():
                    opr = batch_oprs[b][opr_key]
                    if opr.is_first:
                        """for s in alloc_stations:
                        if s.station_key.split('_')[-1] == '0' or \
                            s.station_key.split('_')[-1] == '5802ZP01A0010':
                            stations.append(s)
                            break"""
                        for s in alloc_stations:
                            stations.append(s)
                oprs, opr_stations, opr_links = CaseUtils.add_no_action(n_jobs, n_max_lines, 
                                                                        n_max_oprs, list(set(stations)))
                #print(set(stations))
                batch_oprs[b].update(oprs)
                batch_opr_stations[b].update(opr_stations)
                batch_opr_links[b].update(opr_links)

        if is_save:
            dataset_dir = f'{DIR}/datasets/dev_data/{str(n_jobs)}{str.zfill(str(n_stations),2)}'
            for b in range(batch_size):
                opr_maps, opr_stations, opr_links = batch_oprs[b], batch_opr_stations[b], batch_opr_links[b]
                json_opr_maps = {opr_key: opr.to_json() for opr_key, opr in opr_maps.items()}
                json_opr_stations = {opr_key : [station.to_json() for station in stations] for opr_key, stations in opr_stations.items()}
                json_opr_links = {opr_key : [opr.to_json() for opr in links] for opr_key, links in opr_links.items()}
                save_path = f'{dataset_dir}/{b}i_{n_jobs}j_{n_stations}m.json'
                with open(save_path, 'w') as fp:
                    json.dump({'opr_maps': json_opr_maps, 'opr_stations': 
                            json_opr_stations, 'opr_links': json_opr_links, 'n_max_lines': n_lines[b]}, fp)
                print(f'成功保存数据集 -> {save_path}, 共 {n_lines[b]} 条线(主线和部装线)')
        return batch_oprs, batch_opr_stations, batch_opr_links

    @staticmethod
    def produce_from_file(file_path: str):
        data = JsonUtils.read_json(file_path)
        rows = data['rows']
        
        oprs, opr_stations, opr_links = {}, {}, {}

        def build_orders(orders):
            """获取订单字典
            """
            _orders = {}
            for order in orders:
                _orders[order['makeOrderNum']] = (order['materialCode'],)
            return _orders

        def build_stations(stations):
            """获取工位字典
            """
            def cmp(o1, o2):
                code1 = o1['workStationCode']
                code2 = o2['workStationCode']
                if code1[4:8] == 'ZP01' and code2[4:8] == 'ZP01':
                    return 1 if code1 >= code2 else -1
                if code1[4:8] == 'ZP01':
                    return -1
                if code2[4:8] == 'ZP01':
                    return 1
                return 1 if code1 >= code2 else -1

            _stations = {}
            stations = sorted(stations, key=cmp_to_key(cmp))
            for idx, station in enumerate(stations):
                station_code = station['workStationCode']
                if station_code not in _stations:
                    _stations[station_code] = idx
            return _stations

        def build_oprs(order_oprs):
            """获取订单的所有工序信息
            """
            _oprs = {}
            line_index = 0
            li = set()
            for opr in order_oprs:
                job_idx = opr['makeOrderNum']
                seq = opr['routingSeq']
                opr_no = opr['operationNo']
                station = opr['workStation']
                process_time = opr['workTime']
                ref_opr_seq = opr['refRoutingSeq']
                ref_opr_no = opr['refOperationEnd']
                material = _orders[job_idx][0]
                opr_key = f'OPR_{job_idx}_{seq}_{opr_no}'
                li.add(f'{job_idx}_{seq}')
                ref_opr_key = None
                is_mainline = True
                if ref_opr_no and len(ref_opr_no) > 0:
                    ref_opr_key = f'OPR_{job_idx}_{ref_opr_seq}_{ref_opr_no}'
                    is_mainline = False
                if job_idx not in _oprs:
                    _oprs[job_idx] = collections.defaultdict(list)
                _oprs[job_idx][seq].append((opr_no, opr_key, station, material, process_time, is_mainline, ref_opr_key))
            _stations = {k : v + len(li) for k, v in stations_.items()}
            #print(_stations)
            t = 0
            for i, each_job in enumerate(_oprs.values()):
                each_job = dict(sorted(each_job.items(), key=lambda x : x[0]))
                job_opr_list = []
                is_mainline = True
                vir_station_idx = 0
                vir_station_map = {}
                # 针对主线及所有部装线进行工序链接
                for seq, v in each_job.items():
                    # 为每条线添加一个虚拟首工序
                    vir_opr_key = None
                    vir_t = t
                    vir_station = {}
                    t += 1
                    vir_opr_key = f'VIR_OPR_{i}_{vir_t}'
                    
                    opr_list = []
                    sort_v = sorted(v, key=lambda x : x[1])
                    for item in sort_v:
                        opr_key = item[1]
                        station_idx = _stations[item[2]]
                        station_key = f'STATION_{item[2]}'
                        oprs[opr_key] = Operation(opr_key, item[3], t, i, line_index, {station_key: int(item[4])}, item[5], item[6])
                        opr_list.append(oprs[opr_key])
                        opr_stations[opr_key] = [Station(station_key, None, station_idx)]
                        t += 1
                    #vir_station[f'VIR_STATION_{vir_station_idx}'] = vir_station_idx
                    vir_station[f'VIR_STATION_{line_index}'] = line_index

                    if vir_opr_key is not None:
                        oprs[vir_opr_key] = Operation(vir_opr_key, None, vir_t, i, line_index, vir_station, is_mainline, is_first=True)
                        opr_list.insert(0, oprs[vir_opr_key])
                        """if vir_station_idx in vir_station_map:
                            opr_stations[vir_opr_key] = vir_station_map[vir_station_idx]
                        else:
                            sta = [Station(sk, None, sidx) for sk, sidx in vir_station.items()]
                            opr_stations[vir_opr_key] = sta
                            vir_station_map[vir_station_idx] = sta"""
                        sta = [Station(sk, None, sidx) for sk, sidx in vir_station.items()]
                        opr_stations[vir_opr_key] = sta

                    for k, opr in enumerate(opr_list):
                        if k < len(opr_list)-1:
                            next_opr = opr_list[k+1]
                            opr_links[opr.opr_key] = [next_opr]
                    job_opr_list.append(opr_list)
                    line_index += 1
                    vir_station_idx += 1
                    is_mainline = False
                # 将部装线跟主线对应的工序进行链接  
                for line_oprs in job_opr_list:
                    opr = line_oprs[-1]
                    if not opr.is_mainline and opr.ref_opr_key:
                        opr_key = opr.opr_key
                        # 只需将部装线的最后一道工序与主线对应工序进行链接
                        if opr_key not in opr_links:
                            opr_links[opr_key] = [oprs[opr.ref_opr_key]]
            return line_index

        _orders = build_orders(rows['order']['makeOrderList'])
        stations_ = build_stations(rows['org']['workStationList'])
        line_index = build_oprs(rows['order']['makeOrderOpList'])
        return oprs, opr_stations, opr_links, line_index
    
    @staticmethod
    def add_no_action(n_jobs: int, n_max_lines: int, n_max_oprs: int, stations: List[Station]):
        """增加工序表示不采取任何动作
        Args
        ---------
        n_jobs: 订单数(任务数)
        n_max_lines: 各批次最大线数(主线或部装线)
        n_max_oprs: 各批次最大工序数
        stations: 分配的工位对象
        """
        oprs, opr_stations, opr_links = {}, {}, {}
        opr_key = 'NO_ACT'
        oprs[opr_key] = Operation(opr_key, None, n_max_oprs, n_jobs, n_max_lines, {station.station_key : 1 for station in stations})
        opr_stations[opr_key] = stations
        opr_links[opr_key] = []
        return oprs, opr_stations, opr_links
    
    def load_from_file(file_path: str):
        """加载随机样本文件
        """
        data = JsonUtils.read_json(file_path)
        json_opr_maps = data.get('opr_maps', {})
        json_opr_stations = data.get('opr_stations', {})
        json_opr_links = data.get('opr_links', {})
        n_max_lines = data.get('n_max_lines', -1)
        opr_maps, opr_stations, opr_links = {}, collections.defaultdict(list), collections.defaultdict(list)
        for opr_key, json_opr in json_opr_maps.items():
            opr = Operation(**json_opr)
            opr_maps[opr_key] = opr
            for json_station in json_opr_stations[opr_key]:
                # 忽略calendar配置
                json_station['calendar'] = None
                station = Station(**json_station)
                opr_stations[opr_key].append(station)
        for opr_key, links in json_opr_links.items():
            for link in links:
                next_opr_key = link['opr_key']
                opr_links[opr_key].append(opr_maps[next_opr_key])
        return opr_maps, opr_stations, opr_links, n_max_lines

    @staticmethod
    def random_produce(idx: int, n_orders: int, n_order_oprs: Union[int, List[int]], n_stations: int, 
                            exist_subline=False, is_single_station=False):
        """随机产生样本数据
        Args
        ---------
        idx: 实例索引
        n_orders: 订单数
        n_order_oprs: 各个订单对应的工序数
        n_stations: 工位数
        exist_subline: 是否存在部装线(倒排线)
        is_single_station: 是否单工位(即每个工序只能分配到一个工位)
        mode: 开发集(dev) | 验证集(val) | 测试集(test)
        is_save: 是否保存数据集
        """
        if type(n_order_oprs) not in (int, list):
            raise ValueError('订单工序数必须为整数或者数组')
        if isinstance(n_order_oprs, int):
            # 各个订单的工序数是一样的
            n_order_oprs = [n_order_oprs for _ in range(n_orders)]
        if len(n_order_oprs) != n_orders:
            raise ValueError('订单工序数对应的订单数量需与指定订单数一致')

        proctime_per_opr_min = 5
        proctime_per_opr_max = 20
            
        # 初始化工位日历
        DT_FMT = '%Y-%m-%d %H:%M:%S'
        period1 = Period(datetime.strptime('2023-01-01 08:00:00', DT_FMT),
                                                datetime.strptime('2023-01-01 12:00:00', DT_FMT))
        period2 = Period(datetime.strptime('2023-01-01 14:00:00', DT_FMT),
                                                datetime.strptime('2023-01-01 17:30:00', DT_FMT))
        #print(f'工位可选日历: {period1}, {period2}')
        
        stations = [Station(f'STATION_{i}', Calendar([copy.deepcopy(period1), copy.deepcopy(period2)]), i) for i in range(n_stations)]
        #print(f'创建完成工位数: {len(stations)}')
        
        oprs = []
        opr_links = {}
        line_index = [0]
        n = 0
        for i in range(n_orders):
            order_oprs = []
            for j in range(n_order_oprs[i]):
                # 随机创建工序对象
                opr_key = f'OPR_{i}_{j}'
                material_no = np.random.choice(['A', 'B', 'C'])
                order_oprs.append(Operation(opr_key, material_no, n, i, line_index[0]))
                n += 1
            # 添加工序链接
            opr_links.update(CaseUtils._gen_opr_links(order_oprs, line_index, n_stations, exist_subline))
            oprs.extend(order_oprs)
        # 转换工序对象格式
        opr_maps = {opr.opr_key : opr for opr in oprs}
        #print(f'创建完成订单工序, 订单数: {n_orders}, 工序数: {sum(n_order_oprs)}')

        def alloc_opr_station(opr):
            if opr.is_first:
                return
            # 随机分配的工位数
            n_option_stations = 1
            if is_single_station and len(stations) < len(opr_key.split('_')):
                raise Exception('单机器下工位数必须大于等于工序数')
            choose_stations = []
            if is_single_station:
                choose_stations = [stations[int(opr.opr_key.split('_')[-1])]]
            if not is_single_station:
                n_option_stations = np.random.choice([i for i in range(1, len(stations)+1)])
                choose_stations = list(np.random.choice(stations, size=n_option_stations, replace=False))
            opr_stations[opr.opr_key] = choose_stations
            proc_times_mean = [random.randint(proctime_per_opr_min, proctime_per_opr_max) for _ in range(n_option_stations)]
            for i, m in enumerate(choose_stations):
                alloc_stations.add(m.station_key)
                low_bound = max(proctime_per_opr_min,round(proc_times_mean[i]*(1-0.2)))
                high_bound = min(proctime_per_opr_max,round(proc_times_mean[i]*(1+0.2)))
                #opr.process_time[m.station_key] = np.random.randint(10*60)
                opr.process_time[m.station_key] = random.randint(low_bound, high_bound)

        # 随机为工序分配至少一个工位
        opr_stations = {}
        alloc_stations = set()
        pre_job_index = oprs[0].job_index
        t = 0
        for opr in oprs:
            job_index = opr.job_index
            if job_index != pre_job_index:
                t = 0
            # 为每条线的首工序分配工位
            if opr.is_first:
                machine = stations[t]
                if is_single_station:
                    machine = stations[int(opr.opr_key.split('_')[-1])]
                #print(opr)
                t += 1
                opr_stations[opr.opr_key] = [machine]
                opr.process_time[machine.station_key] = random.randint(1, 20)
                alloc_stations.add(machine.station_key)
        if not is_single_station:
            stations = list(filter(lambda x : x.station_key not in alloc_stations, stations))
        if len(stations) == 0:
            raise Exception('待分配工位不足，请重新设置工位数')
        for opr in oprs[:-1]:
            alloc_opr_station(opr)
        remain_stations = list(filter(lambda x : x.station_key not in alloc_stations, stations))
        # 将剩余未分配的工位分配给最后一道工序
        if len(remain_stations) == 0:
            alloc_opr_station(oprs[-1])
        else:
            if is_single_station:
                remain_stations = remain_stations[:1]
            opr = oprs[-1]
            if not opr.is_first:
                opr_stations[opr.opr_key] = remain_stations
                for m in remain_stations:
                    opr.process_time[m.station_key] = random.randint(1, 20)
        
        return opr_maps, opr_stations, opr_links, line_index[0]

    @staticmethod
    def _gen_opr_links(order_oprs: list, line_index: list, n_stations: int, exist_subline=False):
        """生成工序链接
        """
        opr_links = collections.defaultdict(list)
        def gen_mainline_links(oprs: list):
            """生成主线工序链接
            """
            oprs[0].is_first = True
            for opr in oprs:
                opr.is_mainline = True
                opr.line_index = line_index[0]
            line_index[0] += 1
            for i in range(len(oprs)-1):
                opr_links[oprs[i].opr_key].append(oprs[i+1])
        def gen_subline_links(oprs: list, crs_opr):
            """生成部装线工序链接
            """
            oprs[0].is_first = True
            for opr in oprs:
                opr.line_index = line_index[0]
                opr.is_mainline = False
                opr.ref_opr_key = crs_opr.opr_key
            line_index[0] += 1
            for i in range(len(oprs)-1):
                opr_links[oprs[i].opr_key].append(oprs[i+1])
            opr_links[oprs[-1].opr_key].append(crs_opr)
            
        if len(order_oprs) <= 3 or not exist_subline:
            # 仅生成主线工序
            gen_mainline_links(order_oprs)
            return opr_links
        mainline_ratio = 0.8
        # 主线工序数
        n_mainline_oprs = int(mainline_ratio * len(order_oprs))
        # 部装线工序数
        n_subline_oprs = len(order_oprs) - n_mainline_oprs
        # 主线工序
        mainline_oprs = order_oprs[:n_mainline_oprs]
        mainline_opr_keys = [opr.opr_key for opr in mainline_oprs]
        # 部装线工序
        subline_oprs = list(filter(lambda x : x.opr_key not in mainline_opr_keys, order_oprs))
        max_subline_cnt = min(len(subline_oprs), len(mainline_oprs))
        max_subline_cnt = min(max_subline_cnt, n_stations)
        # 部装线数
        n_subline = int(np.random.randint(1, max_subline_cnt+1, size=1))
        # 生成主线工序链接
        gen_mainline_links(mainline_oprs)
        # 随机产生主线与部装线的交叉点
        rnd_crs_oprs = np.array(mainline_oprs)[np.random.choice([i for i in range(1, n_mainline_oprs)], size=n_subline, replace=True)]
        if n_subline == 1:
            gen_subline_links(subline_oprs, rnd_crs_oprs[0])
            return opr_links
        if n_subline == n_subline_oprs:
            start = 0
            for i in range(n_subline):
                crs_opr = rnd_crs_oprs[i]
                gen_subline_links(subline_oprs[start:start+1], crs_opr)
                start += 1
            return opr_links
        # 随机各个部装线的工序个数
        split_indices = []
        accum = 0
        for i in range(n_subline):
            remain_oprs = n_subline_oprs - accum
            remain_oprs = remain_oprs - (n_subline-i-1)
            rnd_cnt = np.random.choice([i for i in range(1, remain_oprs+1)], size=1, replace=False)[0]
            # 最后一条部装线需要覆盖所有剩余工序
            if i == n_subline-1:
                rnd_cnt = remain_oprs
            split_indices.append(rnd_cnt)
            accum += rnd_cnt
        start = 0
        # 随机生成部装线工序链接
        for i in range(n_subline):
            crs_opr = rnd_crs_oprs[i]
            cnt = int(split_indices[i])
            gen_subline_links(subline_oprs[start:start+cnt], crs_opr)
            start += cnt
        return opr_links