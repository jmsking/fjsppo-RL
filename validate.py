import gym
import torch
import time
import os
import copy
from project.utils.case_utils import CaseUtils

DIR = '/home/bml/storage/chenj1901/schedule_algorithm_demo'

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "{0}/datasets/dev_data/{1}{2}/".format(DIR, env_paras["n_jobs"], str.zfill(str(env_paras["n_stations"]),2))
    #print(file_path)
    #env_paras.update({'source': 'simulation', 'file_path': file_path}) # fjsp
    env_paras.update({'source': 'file', 'file_path': file_path}) # fjsp-po
    batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(**env_paras)
    env = gym.make('as-v0', 
        batch_oprs=batch_oprs, 
        batch_opr_stations=batch_opr_stations, 
        batch_opr_links=batch_opr_links, 
        params=env_paras)
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    print('There are {0} validate instances.'.format(batch_size)) 
    state = env.state
    done = False
    dones = env.done
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, env.memory, is_sample=False, is_train=False)
        idx = state.batch_indices[0]
        print(f'当前时刻 -> {state.batch_time[idx]}, 采取动作: {actions.opr_station_pair[:, 0]}')
        #print(state.batch_opr_features[idx,:,:])
        state, rewards, dones, _ = env.step(actions)
        undone_idx = torch.where(~dones)[0]
        #print(f'未完成批次: {undone_idx}')
        done = dones.all()
    makespan = copy.deepcopy(state.batch_makespan.mean())
    batch_makespan = copy.deepcopy(state.batch_makespan)
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, batch_makespan