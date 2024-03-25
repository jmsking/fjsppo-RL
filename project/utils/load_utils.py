import os
import re
import collections
from project.domain.operation import Operation
from project.domain.station import Station

class LoadUtils:
    """数据加载工具包
    """

    @staticmethod
    def random_produce():
        """随机产生样本
        """
        pass

    @staticmethod
    def load_simulation(file_path: str):
        """加载仿真数据
        """

        def process_lines(lines):
            oprs, opr_stations, opr_links = {}, {}, collections.defaultdict(list)
            # 第一行记录了该批任务中的任务数、机器数及对应的平均加工时长
            #print(len(lines))
            #print(lines[0])
            meta = lines[0].split('\n')[0]
            #print(meta)
            meta = re.split(r"\t|[ ]+", meta)
            #print(meta)
            n_machines = int(meta[1])
            machines = [Station(f'Station_{i}', None, i) for i in range(n_machines)]
            machines_dict = {str(i+1) : machines[i] for i in range(n_machines)}
            opr_idx, line_idx = 0, 0
            for job_idx, line in enumerate(lines[1:]):
                #print(job_idx)
                line = line.split('\n')[0]
                items = re.split(r"\t|[ ]+", line)
                items = list(filter(lambda x : len(x.strip()) > 0, items))
                #print(items)
                if len(items) == 0:
                    continue
                # 每行表示一个任务，第一个元素记录了该任务的工序数
                n_oprs = int(items[0])
                # [p+1, p+q-1]区间记录该工序在可分配的机器上的加工时长
                p = 1
                _oprs = []
                for _ in range(n_oprs):
                    q = int(items[p])*2
                    opr = Operation(f'Opr_{opr_idx}', None, opr_idx, job_idx, line_idx, {})
                    l, r = p+1, p+q
                    choose_machines = []
                    while l <= r:
                        #print(l, r)
                        choose_machines.append(machines_dict[items[l]])
                        opr.process_time[machines_dict[items[l]].station_key] = int(items[l+1])
                        l += 2
                    opr_stations[opr.opr_key] = choose_machines
                    opr_idx += 1
                    _oprs.append(opr)
                    oprs[opr.opr_key] = opr
                    p = r + 1

                for i in range(len(_oprs)-1):
                    opr_links[_oprs[i].opr_key].append(_oprs[i+1])
            #print(oprs)
            #print(opr_stations)
            #print(opr_links)
            #raise Exception()
            return oprs, opr_stations, opr_links


        batch_oprs, batch_opr_stations, batch_opr_links = [], [], []
        #print(file_path)
        #print(file_path)
        for _, _, files in os.walk(file_path):
            #print(len(files))
            for f in files:
                #print(f)
                f = f'{file_path}/{f}'
                #print(f)
                if not os.path.exists(f):
                    continue
                with open(f) as fp:
                    lines = fp.readlines()
                    #print(lines)
                    oprs, opr_stations, opr_links = process_lines(lines)
                    batch_oprs.append(oprs)
                    batch_opr_stations.append(opr_stations)
                    batch_opr_links.append(opr_links)

        return batch_oprs, batch_opr_stations, batch_opr_links

    @staticmethod
    def load_public():
        """加载公开的Benchmark数据
        """
        pass

    @staticmethod
    def load_private():
        """加载私有的真实排产数据
        """
        pass