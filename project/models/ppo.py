import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from project.models.hgnn import GATedge, MLPs
from project.models.mlp import MLPCritic, MLPActor
from project.scheduler.hgnn_scheduler import HGNNScheduler

class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras.get('lr', 0)  # learning rate
        self.betas = train_paras.get('betas', [0, 0])  # default value for Adam
        self.gamma = train_paras.get('gamma', 0)  # discount factor
        self.eps_clip = train_paras.get('eps_clip', 0)  # clip ratio for PPO
        self.K_epochs = train_paras.get('K_epochs', 0)  # Update policy for K epochs
        self.A_coeff = train_paras.get('A_coeff', 0)  # coefficient for policy loss
        self.vf_coeff = train_paras.get('vf_coeff', 0)  # coefficient for value loss
        self.entropy_coeff = train_paras.get('entropy_coeff', 0)  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_opr_station = torch.stack(memory.batch_opr_station, dim=0).transpose(0,1).flatten(0,1)
        old_opr_pre = torch.stack(memory.batch_opr_pre, dim=0).transpose(0, 1).flatten(0, 1)
        old_opr_next = torch.stack(memory.batch_opr_next, dim=0).transpose(0, 1).flatten(0, 1)
        old_opr_features = torch.stack(memory.batch_opr_features, dim=0).transpose(0, 1).flatten(0, 1)
        old_station_features = torch.stack(memory.batch_station_features, dim=0).transpose(0, 1).flatten(0, 1)
        old_edge_features = torch.stack(memory.batch_edge_features, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0,1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1).flatten(0,1)
        old_action_indices = torch.stack(memory.action_indices, dim=0).transpose(0,1).flatten(0, 1)

        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        full_batch_size = old_opr_station.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_opr_station[start_idx: end_idx, :, :],
                                         old_opr_pre[start_idx: end_idx, :, :],
                                         old_opr_next[start_idx: end_idx, :, :],
                                         old_opr_features[start_idx: end_idx, :, :],
                                         old_station_features[start_idx: end_idx, :, :],
                                         old_edge_features[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_indices[start_idx: end_idx])

                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                advantages = rewards_envs[i*minibatch_size:(i+1)*minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = - self.A_coeff * torch.min(surr1, surr2)\
                       + self.vf_coeff * self.MseLoss(state_values, rewards_envs[i*minibatch_size:(i+1)*minibatch_size])\
                       - self.entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                #torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                #torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])