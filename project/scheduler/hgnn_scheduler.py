import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from project.models.hgnn import GATedge, MLPs
from project.models.mlp import MLPCritic, MLPActor
from project.utils.state_utils import StateUtils
from project.simulator.action import Action

class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor
        self.alpha = 0.7
        #self.alpha = 1

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]

        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu))
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)
        self.no_act_steps = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)


    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        batch_opr_features: shape: [len(batch_indices), max(num_opes), in_size_ope]
        batch_station_features: shape: [len(batch_indices), num_mas, in_size_ma]
        batc_edge_features: shape: [len(batch_indices), max(num_opes), num_mas]
    '''
    def get_normalized(self, batch_opr_features, batch_station_features, batch_edge_features, 
                        batch_indices, n_total_opr, is_train=True):
        '''
        :param batch_opr_features: Raw feature vectors of operation nodes
        :param batch_station_features: Raw feature vectors of machines nodes
        :param batch_edge_features: Raw feature vectors of edge
        :param batch_indices: Uncompleted instances
        :param n_total_opr: The number of operations for each instance
        :param is_train: 是否训练模式
        :return: Normalized feats, including operations, machines and edges
        '''
        if is_train:
            mean_opes = torch.mean(batch_opr_features, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(batch_station_features, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(batch_opr_features, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(batch_station_features, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            edge_features_norm = self.feature_normalize(batch_edge_features)  # shape: [len(batch_idxes), num_opes, num_mas]
        else:
            batch_size = batch_indices.size(0)  # number of uncompleted instances
            # There may be different operations for each instance, which cannot be normalized directly by the matrix
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(batch_opr_features[i, :n_total_opr[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(batch_opr_features[i, :n_total_opr[i], :], dim=-2, keepdim=True))
                edge_idxes = torch.nonzero(batch_edge_features[i])
                edge_values = batch_edge_features[i, edge_idxes[:, 0], edge_idxes[:, 1]]
                edge_norm = self.feature_normalize(edge_values)
                batch_edge_features[i, edge_idxes[:, 0], edge_idxes[:, 1]] = edge_norm
            mean_opes = torch.stack(mean_opes, dim=0) # shape: [len(batch_indices), 1, in_size_ope]
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(batch_station_features, dim=-2, keepdim=True) # shape: [len(batch_indices), 1, in_size_ma]
            std_mas = torch.std(batch_station_features, dim=-2, keepdim=True)
            edge_features_norm = batch_edge_features # shape: [len(batch_indices), num_opes, num_mas]
        return ((batch_opr_features - mean_opes) / (std_opes + 1e-5), (batch_station_features - mean_mas) / (std_mas + 1e-5),
                edge_features_norm)

    def get_action_prob(self, state, memory, is_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_indices = state.batch_indices
        # Raw feats
        batch_opr_features = state.batch_opr_features[batch_indices]
        batch_station_features = state.batch_station_features[batch_indices]
        batch_edge_features = state.batch_edge_features[batch_indices]
        # Normalize
        n_total_opr = state.n_total_opr[batch_indices]
        features = self.get_normalized(batch_opr_features, batch_station_features, batch_edge_features, 
                                            batch_indices, n_total_opr, is_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_edge = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_indices), num_mas, out_size_ma]
            h_mas, h_opes = self.get_machines[i](state.batch_opr_station, state.batch_opr_pre, state.batch_indices, features)
            features = (features[0], h_mas, features[2])
            #features = (h_opes, h_mas, features[2])
            # Second Stage, operation node embedding
            # shape: [len(batch_indices), max(num_opes), out_size_ope]
            _h_opes = self.get_operations[i](state.batch_opr_station, state.batch_opr_next, state.batch_opr_pre,
                                            state.batch_indices, features)
            h_opes = self.alpha * h_opes + (1-self.alpha)*_h_opes
            features = (h_opes, features[1], features[2])
            #features = (_h_opes, features[1], features[2])

        batch_size, n_oprs, n_stations = state.batch_opr_proctime.size()

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # shape: [len(batch_indices), out_size_ma]
        if is_train:
            h_opes_pooled = h_opes.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]
        else:
            # There may be different operations for each instance, which cannot be pooled directly by the matrix
            h_opes_pooled = []
            for i in range(len(batch_indices)):
                h_opes_pooled.append(torch.mean(h_opes[i, :n_total_opr[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_indices), d]

        h_opes_padding = h_opes.unsqueeze(-2).expand(-1, -1, n_stations, -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_opes_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_opes_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_opes_padding)

        # Input of actor MLP
        # shape: [len(batch_indices), num_opes, num_mas, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_opes_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        # 获取每个batch可选的动作
        eligible = StateUtils.obtain_ready_actions(state, memory)
        
        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions)

        #print('########')
        #print(scores[0, -5:, :, :])
        #sign_flag = torch.sign(scores)
        #_sign_flag = sign_flag.flatten(1)
        #print(_sign_flag[0, 3800])
        #temperature = torch.pow(memory.temperature[batch_indices], sign_flag)
        #_temperature = temperature.flatten(1)
        #print(_temperature[0, 3800])
        # 保证温度值越大,采取的对应动作的概率越小
        #modify_temperature = torch.where(memory.temperature[batch_indices] > 1e3, -1.0 / temperature, temperature)
        #_modify_temperature = modify_temperature.flatten(1)
        #print(_modify_temperature[0, 3800])
        #temperature = torch.where(sign_flag < 0, temperature, modify_temperature)
        #_t_temperature = temperature.flatten(1)
        #print(_t_temperature[0, 3800])
        #scores /= temperature
        #print(scores[0, -5:, :, :])
        scores = scores.flatten(1)
        mask = eligible[batch_indices].flatten(1)
        scores[~mask] = float('-inf')
        scores /= math.sqrt(self.out_size_ma * 2 + self.out_size_ope * 2)
        action_probs = F.softmax(scores, dim=1)
        nan_mask = torch.isnan(action_probs).any(-1)
        nan_indices = torch.where(nan_mask)[0]
        if nan_indices.size(0) > 0:
            nan_indices = nan_indices[0]
        if nan_mask.any():
            print(nan_indices)
            print(state.batch_opr_features[nan_indices])
            print(state.batch_station_features[nan_indices, :, 1])
            print(state.batch_station_schedule[nan_indices, :, 1])
            print(state.batch_time[nan_indices])
            print(state.batch_done[nan_indices])
            print(state.batch_done.all())
            print(scores[nan_mask][0])
            print(mask[nan_mask][0])
            raise Exception
        #if state.batch_time.cpu() == 55:
        """if not is_train:
            _tmp = torch.where(action_probs[0] > 0)[0]
            print(_sign_flag[0, 640])
            print(_temperature[0, 640])
            print(_modify_temperature[0, 640])
            print(_t_temperature[0, 640])
            print(_tmp)
            print(action_probs[0, _tmp])"""
        # 预测`NO_ACT`步长
        max_v = 100.0
        z = self.no_act_steps(h_pooled)
        #step_size = 1.0 / (1.0 + torch.exp(-z))
        step_size = abs(z)
        step_size *= max_v
        #print(step_size)
        step_size = torch.clamp(step_size, min=0.0, max=max_v)
        step_size = step_size.long()
        step_size = step_size.squeeze()
        state.steps = torch.ones(size=(state.batch_size,))
        state.steps[state.batch_indices] *= step_size
        #print(step_size)

        # Store data in memory during training
        if is_train:
            memory.batch_opr_station.append(copy.deepcopy(state.batch_opr_station))
            memory.batch_opr_pre.append(copy.deepcopy(state.batch_opr_pre))
            memory.batch_opr_next.append(copy.deepcopy(state.batch_opr_next))
            memory.batch_indices.append(copy.deepcopy(state.batch_indices))
            memory.batch_opr_features.append(copy.deepcopy(norm_opes))
            memory.batch_station_features.append(copy.deepcopy(norm_mas))
            memory.batch_edge_features.append(copy.deepcopy(norm_edge))
            memory.eligible.append(copy.deepcopy(eligible))

        return action_probs, step_size

    def act(self, state, memory, is_sample=True, is_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, step_size  = self.get_action_prob(state, memory, is_train)

        # DRL-S, sampling actions following \pi
        if is_sample:
            action_indices = None
            try:
                dist = Categorical(action_probs)
                action_indices = dist.sample()
            except:
                print(action_probs.shape)
                mask = torch.isnan(action_probs).any(-1)
                print(mask.shape)
                print(action_probs[mask][0])
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indices = action_probs.argmax(dim=1)
            #print(action_indices, state.batch_opr_station.size(2))
            #_tmp = action_indices / state.batch_opr_station.size(2)
            #print(_tmp.dtype, (_tmp+1e-7).long())

        eps = 1e-7
        # Calculate the operation and station and job index based on the action index
        oprs = (action_indices / state.batch_opr_station.size(2))
        # 防止float32转long的时候出现 1.0 变成 0
        oprs = (oprs+eps).long()
        stations = (action_indices % state.batch_opr_station.size(2)).long()
        jobs = state.batch_opr_job[state.batch_indices, oprs].long()


        #if not is_train:
        #    print(oprs, stations, jobs)

        #if is_train:
        #    memory.temperature[state.batch_indices, oprs, stations] += 0.001
            #print(memory.temperature.flatten(1)[0])

        # Store data in memory during training
        if is_train:
            memory.states.append(copy.deepcopy(state))
            memory.logprobs.append(dist.log_prob(action_indices))
            memory.action_indices.append(action_indices)

        return Action(torch.stack((oprs, stations, jobs), dim=1)).t()

    def evaluate(self, batch_opr_station, batch_opr_pre, batch_opr_next, 
                batch_opr_features, batch_station_features, batch_edge_features,
                eligible, action_indices):
        batch_indices = torch.arange(0, batch_opr_station.size(-3)).long()
        features = (batch_opr_features, batch_station_features, batch_edge_features)

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            h_mas, h_opes = self.get_machines[i](batch_opr_station, batch_opr_pre, batch_indices, features)
            #features = (h_opes, h_mas, features[2])
            features = (features[0], h_mas, features[2])
            _h_opes = self.get_operations[i](batch_opr_station, batch_opr_next, batch_opr_pre, batch_indices, features)
            h_opes = self.alpha * h_opes + (1-self.alpha)*_h_opes
            features = (h_opes, features[1], features[2])
            #features = (_h_opes, features[1], features[2])

        batch_size, n_oprs, n_stations = batch_opr_station.size()

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_opes_padding = h_opes.unsqueeze(-2).expand(-1, -1, n_stations, -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_opes_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_opes_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_opes_padding)

        # Input of actor MLP
        # shape: [len(batch_idxes), num_oprs, num_mas, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_opes_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        mask = eligible.flatten(1)

        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        scores /= math.sqrt(self.out_size_ma * 2 + self.out_size_ope * 2)
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_indices)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys