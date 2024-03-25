from dataclasses import dataclass
import torch
import copy
import gym
import random
from project.domain.operation import Operation
from project.domain.station import Station
from project.simulator.state import State
from project.simulator.action import Action
from project.simulator.reward import Reward
from project.utils.state_utils import StateUtils
from project.utils.draw_utils import DrawUtils
from project.utils.file_utils import FileUtils
from project.common.memory import Memory
from project.common.storage import Storage
from project.constraints import *

DIR = '/home/bml/storage/chenj1901/schedule_algorithm'

@dataclass
class Env(gym.Env):
    """环境模拟器
    Args
    -----------
    batch_oprs: 工序字典列表
        opr_key (str) -> opr (Operation)
    batch_opr_stations: 工序key对应的该工序可分配的工位列表 
        opr_key (str) -> stations (List[Station])
    batch_opr_links: 工序链接列表
        opr_key (str) -> oprs (List[Operation])
    params: 配置参数
    """
    batch_oprs: list
    batch_opr_stations: list
    batch_opr_links: list
    params: dict
        
    def __post_init__(self):
        self.state = StateUtils.init_state(self.batch_oprs, self.batch_opr_stations, self.batch_opr_links, **self.params)
        self.state0 = copy.deepcopy(self.state)
        om_pairs = StateUtils.obtain_om_pairs(self.batch_oprs, self.batch_opr_stations)
        storage = Storage(self.batch_oprs, self.batch_opr_stations, self.batch_opr_links)
        self.memory = Memory()
        self.memory.update(om_pairs)
        self.memory.register_storage(storage)
        self.init_temperature()
        self.exist_constraint = self.params.get('exist_constraint', False)
        if self.exist_constraint:
            from project.constraints.filter_factory import FilterFactory
            from project.constraints.max_per_day_constraint import MaxPerDayConstraint
            FilterFactory.add('max_per_day_constraint', MaxPerDayConstraint)

    # deprecated
    def init_temperature(self):
        """初始化温度,让`NO_ACT`在开始阶段能更大概率被采样到
        """
        self.memory.temperature = torch.ones(*self.state0.batch_opr_station.size()).unsqueeze(-1)
        n_oprs = self.state0.batch_opr_station.size(1)
        #if self.state0.exist_no_action and self.state0.is_train:
        #    self.memory.temperature[:, n_oprs-1, :] = 0.01

    def step(self, action: Action):
        """某个状态下执行动作后的环境变化
        """
        pre_state = copy.deepcopy(self.state)
        if self.exist_constraint:
            self.memory.record(pre_state, action)
        self.state = StateUtils.state_transfer(self.state, action, memory=self.memory, **self.params)
        return self.state, Reward(pre_state, self.state, action).value, self.done, {}

    def render(self, *, mode=None):
        if not mode or mode not in ('draw',):
            return
        batch_size = self.state.batch_size
        n_jobs = self.state.n_jobs
        if self.state.exist_no_action:
            n_jobs = n_jobs - 1
        n_stations = self.state.n_stations
        color = FileUtils.read_json(f"{DIR}/color_config.json").get("gantt_color", [])
        n_colors = max(n_jobs, n_stations)
        if len(color) < n_colors:
            num_append_color = n_colors - len(color)
            color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
        FileUtils.write_json({"gantt_color": color}, f"{DIR}/color_config.json")
        DrawUtils.draw_station_gantt(batch_size, n_jobs, n_stations, color, 
            self.state.batch_opr_schedule, self.state.n_total_opr, self.memory)
        DrawUtils.draw_job_gantt(batch_size, n_jobs, n_stations, color, 
            self.state.batch_opr_schedule, self.state.n_total_opr, self.memory)

    def reset(self):
        self.state = copy.deepcopy(self.state0)
        return self.state

    def close(self):
        pass
        
    @property
    def done(self):
        """是否已完成
        """
        #self.state.batch_makespan -= 19
        return self.state.batch_done