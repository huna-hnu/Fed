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
        #self.gc1 = gcn(nfeat, nout, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, device):
        super(GCN, self).__init__()

        # self.gc1 = gcn(nfeat, nhid, device)
        # self.gc2 = gcn(nhid, nout, device)
        self.gc1 = gcn(nfeat, nout, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
    #    x = self.gc2(x, adj)
        return F.relu(x)

class Sampling(nn.Module):
    def __init__(self, device):
        super(Sampling, self).__init__()
        self.device=device

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)

        return inputs + rand.to(self.device)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)


class MendGraph(nn.Module):
    def __init__(self, node_len, num_pred, feat_shape, device):
        super(MendGraph, self).__init__()

        self.device =device
        self.num_pred = num_pred  # 预测的消失的邻居节点数
        self.feat_shape = feat_shape  # 特征维度
        self.org_node_len=node_len  # 残缺图的节点数
        self.node_len=self.org_node_len+self.org_node_len*self.num_pred  #修补后的图的节点数， 针对每个残缺图的节点补上num_pred个消失的邻居节点

        for param in self.parameters():
            param.requires_grad=False
    def mend_graph(self,org_feats,org_edges,pred_degree,gen_feats):  # gen_feats: B, layers, N , num_pred*F
        new_edges=[]
        #每个残缺图的节点消失的邻居节点的特征
        gen_feats=gen_feats.view(gen_feats.shape[0],gen_feats.shape[1],gen_feats.shape[2], self.num_pred,self.feat_shape) #B, layers, N, num_pred, F 所有节点的消失节点的特征

       # print('orig feat shape', org_feats.shape)
       ## print('gen feat shape', gen_feats.shape)


        # if config.cuda:
        pred_degree=pred_degree.cpu()
        pred_degree=torch._cast_Int(pred_degree).detach() # N
        org_feats=org_feats.detach()  # B, T, N, F
        fill_feats = torch.cat((org_feats, gen_feats.flatten(2,3)), dim=-2) #  B, layers, N+ N*num_pred, F #  修补后的图的节点特征集
        #print('merge feat shape', fill_feats.shape)
       # print('predicted missing nodes', pred_degree, pred_degree.shape)
        all_nodes = self.org_node_len
        for i in range(self.org_node_len): # 遍历残缺图中的节点
            for j in range(min(self.num_pred,max(0,pred_degree[i]))):
                new_edges.append(np.asarray([i,self.org_node_len+i*self.num_pred+j])) # 修补残缺的边

            all_nodes+=self.num_pred

        #print('all nodes', all_nodes)

        new_edges=torch.tensor(np.asarray(new_edges).reshape((-1,2))) # Em * 2

        new_edges=new_edges.to(self.device)
        if len(new_edges)>0:
            fill_edges=torch.cat((org_edges,new_edges), dim =0 ) # E+Em, 2 修补后的图的边集
        else:
            fill_edges=torch.clone(org_edges)
        return fill_edges,fill_feats, all_nodes  # B, layers, N+ N*num_pred, F

    def get_adj(self,edges, nodes):
        # if config.cuda:
        #         #     edges=edges.cpu()
        #         # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #         #                          shape=(self.node_len, self.node_len),
        #         #                          dtype=np.float32)  # 邻接矩阵， 正常格式
        #         #
        #         # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #         #
        #         # adj = self.normalize(adj + sp.eye(adj.shape[0]))
        #         # #adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        #         # # if config.cuda:
        #         # #     adj=adj.cuda()
        mG=nx.Graph()

        edges =np.asarray( edges.cpu())

       # print('nodes of G', len(list(mG.nodes())))

        for i in range(nodes):
            mG.add_node(i)     # global graph
           # print(i)

      #  print('nodes', nodes)
       # print(edges, edges.shape)



        for i in range(edges.shape[0]):
            mG.add_edge(edges[i][0], edges[i][1])


       # print('nodes of G', len(list(mG.nodes())))

        adj = np.asarray(nx.adjacency_matrix(mG).todense())

        #print('adj shape', adj.shape)

        adj = self.normalize(adj + sp.eye(adj.shape[0]))

        return torch.Tensor(adj)

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self,org_feats,org_edges, pred_missing,gen_feats):
        fill_edges,fill_feats, fill_nodes_num=self.mend_graph(org_feats,org_edges,pred_missing,gen_feats) #B, layers, N+ N*num_pred, F
        adj=self.get_adj(fill_edges, fill_nodes_num)
        return fill_feats,adj

class LocalSage_Plus(nn.Module):

    def __init__(self, args,  feat_shape,  node_len,device):
        super(LocalSage_Plus, self).__init__()

        self.device= device
        self.encoder_model = GNN(nfeat=feat_shape,  # 输入特征维度    两层GCN加中间的dropout层
                                 nhid=args.hidden,  # 隐藏层维度 32
                                 nout= args.latent_dim,  # 128
                                 dropout=args.dropout_m, device=device)  # 0.5
        self.reg_model = RegModel(in_dim= args.latent_dim, device=device )  # 线性转换， 输入维度为latent_dim, 输出维度为1
        self.gen = Gen(dropout=args.dropout_m, num_pred=args.num_pred, in_dim=args.latent_dim, feat_shape=feat_shape, device=device) # 多层线性转换加激活层，生成要预测的节点特征
        self.mend_graph=MendGraph(node_len=node_len, num_pred=args.num_pred, # node_len 残缺图节点数
                                  feat_shape=feat_shape, device=device) # node_ids 残缺图所有节点

        self.gcn= GCN(nfeat=feat_shape,    # 提取修补图中的空间特征 两层GCN加中间的dropout层
                            nhid=args.hidden,# 中间隐藏维度
                            nout= feat_shape,
                            dropout=args.dropout, device=device)

    def forward(self, feat, edges,adj):  # N, B, T/layers, F 输入为残缺图的节点特征，边和邻接矩阵
        feat= feat.permute(1,2,0,3) # B, T/layers, N, F
        x = self.encoder_model(feat, adj) #  # B, layers, N, F
        degree = self.reg_model(x)  # 1*N 生成残缺图中每个节点预测的消失节点数
        gen_feat = self.gen(x)  # B, layers, N , num_pred*F 残缺图中所有节点生成预测的消失的节点的特征
        mend_feats,mend_adj=self.mend_graph(feat,edges, degree,gen_feat) # B, layers, N+N*num_pred, F图修补
      #  print('adj shape', mend_adj.shape)
      #  print('feature shape', mend_feats.shape)
        gcn_f=self.gcn(mend_feats, mend_adj.to(self.device))  # B, layers, N+N*num_pred, F 特征提取器
        return degree, gen_feat,gcn_f.permute(2, 0,1,3) # N+N*num_pred, B, layers,  F


class Gen(nn.Module):
    def __init__(self, dropout,num_pred,in_dim, feat_shape, device):
        super(Gen, self).__init__()
        self.num_pred=num_pred
        self.feat_shape=feat_shape
        self.sample = Sampling(device)

        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,1024)
        self.fc_flat = nn.Linear(1024, self.num_pred * self.feat_shape)

        self.dropout = dropout

    def forward(self, x):  # B, layers, N, F
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x #B, layers, N, self.num_pred * self.feat_shape

class RegModel(nn.Module):
    def __init__(self,in_dim, device):
        super(RegModel,self).__init__()
        self.reg_1 = nn.Linear(in_dim,1)

    def forward(self,x): # # B, layers, N, F
        x = self.reg_1(x).flatten(0,1) # B*layers, N, 1

        x = F.relu(torch.mean(x, dim=0))
        return x  # N


class FedSage_Plus(nn.Module):
    def __init__(self, local_graph:LocalSage_Plus):
        super(FedSage_Plus, self).__init__()
        self.encoder_model =local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph=local_graph.mend_graph
        self.gcn=local_graph.gcn

        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.gcn.requires_grad_(False)

    def forward(self, feat, edges,adj):
        feat = feat.permute(1, 2, 0, 3)  # B, T/layers, N, F
        x = self.encoder_model(feat, adj)  # # B, layers, N, F
        degree = self.reg_model(x)  # 1*N 生成残缺图中每个节点预测的消失节点数
        gen_feat = self.gen(x)  # B, layers, N , num_pred*F 残缺图中所有节点生成预测的消失的节点的特征
        mend_feats, mend_adj = self.mend_graph(feat, edges, degree, gen_feat)  # B, layers, N+N*num_pred, F图修补
        #  print('adj shape', mend_adj.shape)
        #  print('feature shape', mend_feats.shape)
        gcn_f = self.gcn(mend_feats, mend_adj.to(self.device))  # B, layers, N+N*num_pred, F 特征提取器
        return degree, gen_feat, gcn_f.permute(2, 0, 1, 3)  # N+N*num_pred, B, layers,  F

