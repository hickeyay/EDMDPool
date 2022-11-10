import torch
import torch.nn as nn
import numpy as np
import math
import networkx as nx
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import *
from torch_geometric.data import Data
from networkx.classes.function import *
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv

class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.2):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        self.dense = nn.Linear(hidden_size, 1)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)

        return hidden_states


class EDMDPool(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):

        super(EDMDPool, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h

        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)

        h = self.bottom_gcn(g, h)

        for i in range(self.l_n):

            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])

            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()

        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GATConv(in_dim, out_dim)
        self.conv3 = SAGEConv(in_dim, out_dim)
        self.conv4 = ChebConv(in_dim, out_dim, 2)

    def forward(self, g, h):
        h = self.drop(h)
        x = h.float()
        edge_index, _ = dense_to_sparse(g)
        h = F.relu(self.conv1(x, edge_index))
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

        self.view_att = Parameter(torch.Tensor(2, 2))
        nn.init.xavier_uniform_(self.view_att.data)
        self.view_bias = Parameter(torch.Tensor(2))
        nn.init.zeros_(self.view_bias.data)

    def forward(self, g, h):

        h_att = h
        h_att = h_att.unsqueeze(0)
        input_size = h_att.size(-1)
        hidden_size = 128
        num_heads = 2
        hidden_dropout_prob = 0.1
        attention = SelfAttention(num_heads, input_size, hidden_size, hidden_dropout_prob).cuda()
        attout = attention(h_att)
        x_scores1 = attout.squeeze(0)
        score1 = torch.sigmoid(x_scores1).view(-1, 1)

        A = g.cpu()
        g_for_score = nx.from_numpy_matrix(A.numpy())
        betweenness_centrality = list(nx.betweenness_centrality(g_for_score).values())
        degree_centrality = list(nx.degree_centrality(g_for_score).values())
        closeness_centrality = list(nx.closeness_centrality(g_for_score).values())
        closeness_centrality = torch.Tensor(closeness_centrality).cuda()
        betweenness_centrality = torch.Tensor(betweenness_centrality).cuda()
        degree_centrality = torch.Tensor(degree_centrality).cuda()

        x_scores2 = betweenness_centrality + degree_centrality + closeness_centrality

        score2 = torch.sigmoid(x_scores2).view(-1, 1)

        score_cat = torch.cat([score1, score2], dim=-1)
        max_value, _ = torch.max(torch.abs(score_cat), dim=0)
        score_cat = score_cat / max_value

        score_weight = torch.sigmoid(torch.matmul(score_cat, self.view_att) + self.view_bias)
        score_weight = torch.softmax(score_weight, dim=1)

        scores = torch.sigmoid(torch.sum(score_cat * score_weight, dim=1))

        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):

    num_nodes = g.shape[0]

    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
