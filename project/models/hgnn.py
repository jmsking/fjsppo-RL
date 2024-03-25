import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
    
class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats # embedding size

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, 1, num_head * out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, 1, num_head * out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, 1, num_head * out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_semantic = nn.Parameter(torch.rand(size=(1, 1, num_head * out_feats), dtype=torch.float))
        self.W_semantic = nn.Linear(out_feats*num_head, out_feats*num_head, bias=True)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, batch_opr_station, batch_opr_pre, batch_indices, feat):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        #   W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        assert isinstance(feat, tuple), 'features include opr, station and edge feature'
        
        h_src = self.feat_drop(feat[0]) # opr
        h_dst = self.feat_drop(feat[1]) # machine
        if not hasattr(self, 'fc_src'):
            self.fc_src, self.fc_dst = self.fc, self.fc
        feat_src = self.fc_src(h_src)
        feat_dst = self.fc_dst(h_dst)

        feat_edge = self.fc_edge(feat[2].unsqueeze(-1))
        #print(feat[0].shape, feat[1].shape, feat[2].shape)
        #print(h_src.shape, h_dst.shape)
        #print(feat_src.shape, feat_dst.shape, feat_edge.shape)

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)
        el_add_ee = batch_opr_station[batch_indices].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + batch_opr_station[batch_indices].unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a) # (batch_size, n_oprs, n_machines, 1)
        ekk = self.leaky_relu(er + er) # (batch_size, n_machines, 1)
        el_add_el = batch_opr_pre[batch_indices].unsqueeze(-1) * el.unsqueeze(-2) + el.unsqueeze(-2)
        ejl = self.leaky_relu(el_add_el) # (batch_size, n_oprs, n_oprs, 1)
        ell = self.leaky_relu(el + el) # (batch_size, n_oprs, 1)
        #print(el.shape, er.shape, ee.shape, el_add_ee.shape, a.shape, eijk.shape, ekk.shape, ell.shape)

        # Normalize attention coefficients
        # shape -> (batch_size, n_oprs, n_machines, 1)
        mask_opr_machine = torch.cat((batch_opr_station[batch_indices].unsqueeze(-1)==1,
                          torch.full(size=(batch_opr_station[batch_indices].size(0), 1,
                                           batch_opr_station[batch_indices].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        # shape -> (batch_size, n_machines, n_oprs, 1)
        batch_station_opr = batch_opr_station.permute(0, 2, 1)
        mask_machine_opr = torch.cat((batch_station_opr[batch_indices].unsqueeze(-1)==1,
                          torch.full(size=(batch_station_opr[batch_indices].size(0), 1,
                                            batch_station_opr[batch_indices].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        # shape -> (batch_size, n_oprs, n_oprs, 1)
        mask_opr_opr = torch.cat((batch_opr_pre[batch_indices].unsqueeze(-1)==1,
                          torch.full(size=(batch_opr_pre[batch_indices].size(0), 1,
                                            batch_opr_pre[batch_indices].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        e[~mask_opr_machine] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        #print((alpha_kk.squeeze().unsqueeze(-1)).shape)
        #print(feat_dst.shape)
        nu_k_prime = torch.sigmoid(b+c)
        #print(a.shape, b.shape, c.shape, nu_k_prime.shape)
        
        # Calculate an return operation embedding (first meta-path: machine -> opr)
        ekij = eijk.permute(0, 2, 1, 3) # (batch_size, n_machines, n_oprs, 1)
        e = torch.cat((ekij, ell.unsqueeze(-3)), dim=-3)
        e[~mask_machine_opr] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_kij = alpha[..., :-1, :]
        alpha_ll = alpha[..., -1, :].unsqueeze(-2)
        Wmu_kij = feat_edge.permute(0,2,1,3) + feat_dst.unsqueeze(-2)
        a = Wmu_kij * alpha_kij.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_src * alpha_ll.squeeze().unsqueeze(-1)
        #print((alpha_ll.squeeze().unsqueeze(-1)).shape)
        #print(feat_src.shape)
        #raise Exception()
        mu_ij_prime_1 = torch.sigmoid(b+c) # shape: (batch_size, n_oprs, out_feats)

        # Calculate an return operation embedding (second meta-path: opr -> opr)
        e = torch.cat((ejl, ell.unsqueeze(-3)), dim=-3)
        e[~mask_opr_opr] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_jl = alpha[..., :-1, :]
        alpha_ll = alpha[..., -1, :].unsqueeze(-2)
        Wm_k = feat_src.unsqueeze(-2)
        a = Wm_k * alpha_jl.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_src * alpha_ll.squeeze().unsqueeze(-1)
        mu_ij_prime_2 = torch.sigmoid(b+c)
        #print(a.shape, b.shape, c.shape, mu_ij_prime.shape)
        #raise Exception()
    
        # Calculate semantic-level operation embedding
        meta_path_coeff_1 = torch.tanh(self.W_semantic(mu_ij_prime_1))
        meta_path_coeff_2 = torch.tanh(self.W_semantic(mu_ij_prime_2))
        e_path_1 = (meta_path_coeff_1 * self.attn_semantic).sum(dim=-1).mean(dim=-1).unsqueeze(-1)
        e_path_2 = (meta_path_coeff_2 * self.attn_semantic).sum(dim=-1).mean(dim=-1).unsqueeze(-1)
        e_path = torch.cat((e_path_1, e_path_2), dim=-1)
        alpha = F.softmax(e_path, dim=-1)
        mu_ij_prime = alpha[:, 0].reshape(-1, 1, 1) * mu_ij_prime_1 + alpha[:, 1].reshape(-1, 1, 1) * mu_ij_prime_2
        #print(mu_ij_prime.shape)
        #raise Exception
        
        return nu_k_prime, mu_ij_prime
        

class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        b = torch.sum(a, dim=-2)
        c = self.project(b)
        return c

class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, batch_opr_station, ope_pre_adj_batch, ope_sub_adj_batch, batch_indices, feats):
        '''
        :param batch_opr_station: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_indices: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_indices])

        # Calculate an return operation embedding
        adj = (batch_opr_station[batch_indices], ope_pre_adj_batch[batch_indices],
               ope_sub_adj_batch[batch_indices], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime