import numpy as np
import torch
import gym
from project.simulator.env import Env
from project.utils.state_utils import StateUtils
from project.scheduler.base_scheduler import BaseScheduler
from project.utils.case_utils import CaseUtils

params = {
    'batch_size': 2,
    'n_jobs': 2,
    'n_stations': 5,
    'opr_feat_dim': 6,
    'station_feat_dim': 3,
    'is_save': False,
    'no_action': False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
if device.type == 'cuda':
    torch.cuda.set_device(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print("PyTorch device: ", device.type)
torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

params['device'] = device

batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(**params)

# 创建环境并初始化
env = gym.make('as-v0', batch_oprs=batch_oprs, 
        batch_opr_stations=batch_opr_stations, 
        batch_opr_links=batch_opr_links, 
        params=params)
print('num_job: ', params['n_jobs'], '\tnum_mas: ', params['n_stations'], '\tnum_oprs ', len(batch_oprs[0]))

scheduler = BaseScheduler(env, **params)

scheduler.start()