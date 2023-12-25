from __future__ import print_function, division

import networkx as nx
from torch import optim
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np


from fedmendgcn import LocalSage_Plus

import torch.nn as nn



class MendGCN(nn.Module):  # local client
    def __init__(self, args, subGraph, in_size, nodes_per_client, device):
        super(MendGCN, self).__init__()

        self.args = args

        self.device = device

        self.subG = subGraph  #subGraph.hasG  # 完整子图

        self.node_subG_ids = list(self.subG.nodes())  # 完整子图节点集，列表

        self.feat_shape = in_size  # 特征维度


        self.n_nodes =len(list(self.node_subG_ids)) #len(list(self.hasG_hide.nodes()))  # 残缺图中节点的个数

        self.num_pred = args.num_pred  # 预测数目


        self.neighgen = LocalSage_Plus(args, feat_shape=self.feat_shape, node_len=self.n_nodes, nodes_per_client=nodes_per_client, device=device
                                       )  # client model，图修补，

        adj = np.asarray(nx.adjacency_matrix(self.subG).todense())
        # print(adj.shape)
        adj= adj+ np.eye(adj.shape[0])
        self.adj = torch.Tensor(adj).to( self.device)    # 残缺图的邻接矩阵




    def forward(self, node_features): # N, B，layers， F

        #self.get_train_test_feat_targets(node_features) # 提取缺失的邻居节点数， 节点特征

        input_feat = node_features # 残缺图的节点特征 N, B, T/layers, F
        input_adj = self.adj  # 残缺图的邻接矩阵

        output_missing, output_missfeat, mend_gfeat = self.neighgen(input_feat, input_adj) ## N+N*num_pred, B, layers,  F 计算client模型得到分类结果

        return mend_gfeat # N, B, Layers, F 完整子图节点特征



