import numpy as np
import torch
import time
import json
from project.simulator.state import State
from project.simulator.action import Action
from project.simulator.env import Env
from project.utils.state_utils import StateUtils
from project.models.ppo import PPO

DIR = '/chenj1901/schedule_algorithm'

def get_model(**param):
    # Load config and init objects
    with open(f"{DIR}/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    env_paras.update(param)
    model_paras.update(param)

    model = PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    return model.policy_old

class BaseScheduler:
    """基础调度器
    Args
    --------
    env: 仿真环境
    kwargs: 额外参数
    """
    def __init__(self, env, **kwargs):
        """初始化
        """
        self.env = env
        self.kwargs = kwargs
        
    def choose_action(self, state, policy=None):
        """根据当前状态选择动作
        """
        if policy:
            with torch.no_grad():
                return policy.act(state, self.env.memory)
        # 可选动作集
        eligible = StateUtils.obtain_ready_actions(state, self.env.memory)
        #print(eligible)
        self.env.memory.eligible.append(eligible)
        batch_size = eligible.size(0)
        oprs, stations = [], []
        for b in range(batch_size):
            arg_indices = np.argwhere(eligible[b,...].cpu()).to(state.device)
            _oprs = arg_indices[0]
            _stations = arg_indices[1]
            oprs.append(_oprs[0])
            stations.append(_stations[0])
        # 采用随机策略
        oprs = torch.stack(oprs, dim=0).long()
        stations = torch.stack(stations, dim=0).long()
        jobs = state.batch_opr_job.gather(1, oprs.unsqueeze(-1)).squeeze(-1).long()
        return Action(torch.stack((oprs, stations, jobs), dim=1)).t()
            
    def start(self):
        """开始调度
        """
        state = self.env.state
        done = False
        dones = self.env.done
        last_time = time.time()
        print(state.batch_opr_next)
        print(state.batch_opr_proctime)
        policy = get_model(**self.kwargs)
        #policy = None
        # Schedule in parallel
        while ~done:
            actions = self.choose_action(state, policy)
            print(f'当前时刻 -> {state.batch_time}, 采取动作: {actions.opr_station_pair}')
            state, rewards, dones, _ = self.env.step(actions)
            """print(state.batch_opr_features)
            print(state.batch_station_features)
            print(state.batch_makespan)
            print(rewards)"""
            
            done = dones.all()
        print("spend_time: ", time.time()-last_time)
        """print(rewards)
        print(state.batch_station_features)
        print(state.batch_opr_features)
        print(state.batch_makespan)
        print(state.batch_time)"""
        #env.render(mode='draw')
        self.env.reset()
        print('完成排产')