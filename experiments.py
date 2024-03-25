import copy
import json
import os
import random
import time
import argparse

import gym
import pandas as pd
import torch
import numpy as np

#import pynvml
from project.models.ppo import PPO
from project.utils.case_utils import CaseUtils
from project.common.constant import DIR

benchmark = {
    'exp_1': 111.67, 'exp_2': 211.22, 'exp_3': 166.92, 'exp_4': 215.78, 'exp_5': 313.04, 'exp_6': 416.18,
    'exp_7': 201.00, 'exp_8': 1030.83, 'exp_9': 1187.48, 'exp_10': 955.90,
    'exp_11': 443, 'exp_12': 259542, 'exp_13': 580, 'exp_14': 259443
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    setup_seed(1024)
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    #pynvml.nvmlInit()
    #handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    best_ckpt = 'opt/save_best_10_5_20'

    with open(f'{DIR}/config.json', 'r') as load_f:
        load_dict = json.load(load_f)
    #print(load_dict)
    _bc = '_'.join(best_ckpt.split('_')[:-1])
    with open(f'{DIR}/ckpts/{_bc}_paras.json', 'r') as load_f:
        best_paras = json.load(load_f)

    load_dict['model_paras'].update(best_paras)
    load_dict['train_paras'].update(best_paras)

    print(f'Loading config_experiments_{args.exp}.json')
    # Load config and init objects
    with open(f"{DIR}/config_experiments_{args.exp}.json", 'r') as load_f:
        exp_dict = json.load(load_f)
    load_dict.update(exp_dict)
    model_paras = load_dict["model_paras"]
    test_paras = load_dict["test_paras"]
    test_paras["device"] = device
    test_paras['opr_feat_dim'] = load_dict['env_paras']['opr_feat_dim']
    test_paras['station_feat_dim'] = load_dict['env_paras']['station_feat_dim']
    test_paras['is_train'] = False
    model_paras["device"] = device
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    model = PPO(model_paras, test_paras, num_envs=test_paras["batch_size"])

    best_ckpt_path = f'{DIR}/ckpts/{best_ckpt}.pt'
    if device.type == 'cuda':
        model_ckpt = torch.load(best_ckpt_path)
    else:
        model_ckpt = torch.load(best_ckpt_path, map_location='cpu')
    print(f'\nloading checkpoint success {best_ckpt_path}')
    model.policy.load_state_dict(model_ckpt)

    batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(is_same_opr=True, **test_paras)
    env = gym.make('as-v0', 
        batch_oprs=batch_oprs, 
        batch_opr_stations=batch_opr_stations, 
        batch_opr_links=batch_opr_links, 
        params=test_paras)
    
    start = time.time()
    batch_size = test_paras["batch_size"]
    print('There are {0} test instances.'.format(batch_size)) 
    #print('num_oprs ', len(batch_oprs[0]))
    state = env.state
    done = False
    dones = env.done
    while ~done:
        with torch.no_grad():
            actions = model.policy.act(state, env.memory, is_sample=False, is_train=False)
        #n_scheduled = sum(state.batch_opr_features[0, :, 0])
        #print(f'当前时刻 -> {state.batch_time[0]}, 采取动作: {actions.opr_station_pair[:, 0]} [{n_scheduled}]')
        state, rewards, dones, _ = env.step(actions)
        done = dones.all()
        #print(state.batch_opr_features[0])
    makespan = copy.deepcopy(state.batch_makespan.mean())
    #batch_makespan = copy.deepcopy(state.batch_makespan)
    #env.render(mode='draw')
    env.reset()
    #print('total test time: ', time.time() - start, 'avg makespan: ', makespan, '\n')
    diff = makespan - benchmark[f'exp_{args.exp}']
    print('total test time: ', time.time() - start, 'makespan diff: ', makespan, diff, '\n')
    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AR-HGAT model')
    parser.add_argument('exp', type=int, help='Experiement name')
    args = parser.parse_args()
    for i in range(1):
        diff = main(args)
        if diff < 0:
            print(f'####################### {diff}')