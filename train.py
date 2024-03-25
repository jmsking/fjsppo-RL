import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

from project.models.ppo import PPO
from project.utils.case_utils import CaseUtils
from project.simulator.env import Env
from project.utils.state_utils import StateUtils
from project.common.constant import DIR
from validate import validate, get_validate_env

import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(learning_rate, optimizer, epoch, max_iterations, learning_rate_decay = 0.98):
    learning_rate = learning_rate * (learning_rate_decay ** epoch)
    #learning_rate = learning_rate * (1 - epoch/max_iterations)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return learning_rate

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    #device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open(f"{DIR}/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    env_valid_paras["is_train"] = False

    num_jobs = env_paras["n_jobs"]
    num_mas = env_paras["n_stations"]

    ckpt_save_path = f'{DIR}/ckpts'
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')
    best_paras = {}
    best_paras.update(model_paras)
    best_paras.update(train_paras)
    best_paras.pop('device')

    paras_file = '{0}/save_best_{1}_{2}_paras.json'.format(ckpt_save_path, num_jobs, num_mas)
    with open(paras_file, 'w') as f:
        #print(best_paras)
        json.dump(best_paras, f, indent=4)

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    model = PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    valid_env = get_validate_env(env_valid_paras)  # Create an environment for validation
    # stop threshold
    pre_loss = 1e10
    loss = 0
    loss_threshold = -1e-3
    # Start training iteration
    start_time = time.time()
    env = None
    train_loss, valid_rewards = [], []
    for i in range(1, train_paras["max_iterations"]+1):
        """if abs(pre_loss - loss) <= loss_threshold:
            print('损失不再变化!!!')
            break"""
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(**env_paras)
            env = gym.make('as-v0', 
                batch_oprs=batch_oprs, 
                batch_opr_stations=batch_opr_stations, 
                batch_opr_links=batch_opr_links, 
                params=env_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_oprs ', len(batch_oprs[0]))

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done
        last_time = time.time()
        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, env.memory)
            #print(f'当前时刻 -> {state.batch_time[0]}, 采取动作: {actions.opr_station_pair[:, 0]}')
            state, rewards, dones, _ = env.step(actions)
            #print(f'获取奖励值: {rewards[0]}')
            done = dones.all()
            env.memory.rewards.append(rewards)
            env.memory.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time()-last_time)
        #env.render(mode='draw')
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            pre_loss = loss
            loss, reward = model.update(env.memory, env_paras, train_paras)
            train_loss.append(loss)
            valid_rewards.append(reward)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            env.memory.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 20 in paper)
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            avg_result, batch_results = validate(env_valid_paras, valid_env, model.policy_old)

            # Save the best model
            if avg_result < makespan_best:
                makespan_best = avg_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(ckpt_save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([avg_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

            #adjust_learning_rate(model.lr, model.optimizer, i, train_paras["max_iterations"])

    print("total_time: ", time.time() - start_time)
    print('makespan: ', makespan_best)

    plt.title('train loss curve')
    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.show()
    plt.savefig(f'{DIR}/ckpts/curve.jpg')
    

if __name__ == '__main__':
    main()