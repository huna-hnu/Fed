import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution, gcn

import torch
import numpy as np
import scipy.sparse as sp

class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, device):
        super(GNN, self).__init__()

        self.gc1 = gcn(nfeat, nhid, device)
        self.gc2 = gcn(nhid, nout, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, device):
        super(GCN, self).__init__()

        self.gc = gcn(nfeat, nout, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return F.relu(x)


class MendGraph(nn.Module):
    def __init__(self, node_len, num_pred, feat_shape, nodes_per_client, device,use_att_p=False, use_att_N=False):
        super(MendGraph, self).__init__()

        self.device =device
        self.num_pred = num_pred  # 预测的消失的邻居节点数
        self.feat_shape = feat_shape  # 特征维度
        self.org_node_len=node_len  # 残缺图的节点数
        self.node_len=self.org_node_len+self.org_node_len*self.num_pred  #修补后的图的节点数， 针对每个残缺图的节点补上num_pred个消失的邻居节点
        self.nodes_per_client=nodes_per_client

        self.MLP= nn.Linear(nodes_per_client*num_pred, nodes_per_client)


    def mend_graph(self,org_feats,gen_feats):  # gen_feats: B, layers, N , num_pred*F
        new_edges=[]
        #每个残缺图的节点消失的邻居节点的特征
        gen_feats=gen_feats.view(gen_feats.shape[0],gen_feats.shape[1],gen_feats.shape[2], self.num_pred,self.feat_shape) #B, layers, N, num_pred, F 所有节点的消失节点的特征

        # if config.cuda:
        org_feats=org_feats.detach()  # B, T, N, F

        fill_feats = gen_feats.flatten(2, 3)  # B, layers, N*num_pred, F

        fill_feats=fill_feats.transpose(2, 3) #B, layers, F, N*num_pred
        if self.org_node_len < self.nodes_per_client:
            fill_feats = F.pad(fill_feats, (0, int(self.num_pred) * (self.nodes_per_client - self.org_node_len)))


        fill_feats = self.MLP(fill_feats)[:,:,:,0:self.org_node_len].transpose(2,3) # B, layers, N, F


        return fill_feats+org_feats #fill_feats.transpose(2,3)  # B, layers, N+ N*num_pred, F


    def forward(self,org_feats,gen_feats):
        fill_feats=self.mend_graph(org_feats, gen_feats) #B, layers, N+ N*num_pred, F
        return fill_feats

class LocalSage_Plus(nn.Module):

    def __init__(self, args,  feat_shape,  node_len, nodes_per_client, device):
        super(LocalSage_Plus, self).__init__()

        self.device= device
        self.encoder_model = GNN(nfeat=feat_shape,  # 输入特征维度    两层GCN加中间的dropout层
                                 nhid=args.hidden,  # 隐藏层维度 32
                                 nout= args.latent_dim,  # 128
                                 dropout=args.dropout_m, device=device)  # 0.1

        self.gen = Gen(dropout=args.dropout_m, num_pred=args.num_pred, in_dim=args.latent_dim, feat_shape=feat_shape, device=device) # 多层线性转换加激活层，生成要预测的节点特征
        self.mend_graph=MendGraph(node_len=node_len, num_pred=args.num_pred, # node_len 残缺图节点数
                                  feat_shape=feat_shape, nodes_per_client=nodes_per_client, device=device)#, use_att_p=args.use_att_p, use_att_N=args.use_att_N) # node_ids 残缺图所有节点

        self.gcn= GCN(nfeat=feat_shape,    # 提取修补图中的空间特征 两层GCN加中间的dropout层
                            nhid=args.hidden,# 中间隐藏维度
                            nout= feat_shape,
                            dropout=args.dropout, device=device)

    def forward(self, feat, adj):  # N, B, T/layers, F 输入为残缺图的节点特征，边和邻接矩阵
        feat = feat.permute(1, 2, 0, 3)  # B, T/layers, N, F
        x = self.encoder_model(feat, adj)  # # B, layers, N, F
        gen_feat = self.gen(x)  # B, layers, N , num_pred*F 残缺图中所有节点生成预测的消失的节点的特征
        mend_feats = self.mend_graph(feat, gen_feat)  # B, layers, N, F图修补  线性转换
        #  print('adj shape', mend_adj.shape)
        #  print('feature shape', mend_feats.shape)
        gcn_f = self.gcn(mend_feats, adj.to(self.device))  # B, layers, N, F 特征提取器
        return 0, gen_feat, gcn_f.permute(2, 0, 1, 3)  # N+N*num_pred, B, layers,  F


class Gen(nn.Module):
    def __init__(self, dropout,num_pred,in_dim, feat_shape, device):
        super(Gen, self).__init__()
        self.num_pred=num_pred
        self.feat_shape=feat_shape


        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,1024)
        self.fc_flat = nn.Linear(1024, self.num_pred * self.feat_shape)

        self.dropout = dropout

    def forward(self, x):  # B, layers, N, F

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x #B, layers, N, self.num_pred * self.feat_shape




