import copy
import math

from copy import deepcopy
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from MendGCN import MendGCN

from utils.utils import load_adj, DataLoader

#from models import GRUSeq2SeqWithGraphNet
from models.GCNSeq2SeqM import GRUSeq2SeqWithGraphNet
from utils.utils import unscaled_metrics

#from utils import config
import networkx as nx
from utils import utils

import itertools


class FedNodePredictorClient(nn.Module):
    def __init__(self, client_params, device):
        super().__init__()
        self.args = client_params['args']

        self.optimizer_name = client_params['args'].Optimizer_name  #Adam
        self.train_dataset = client_params['train_dataset']  #
        self.val_dataset = client_params['val_dataset']
        self.test_dataset = client_params['test_dataset']
        self.feature_scaler = client_params['feature_scaler']
        self.sync_every_n_epoch = self.args.sync_every_n_epoch
        self.lr = self.args.lr
        self.weight_decay = self.args.weight_decay
        self.batch_size = self.args.batch_size
        self.base_model_kwargs = self.args

        self.subGraph= client_params['subGraphs']
        self.client_i = client_params['client_i']
        self.nodes_per_client=client_params['nodes_per_client']
        start_global_step= client_params['start_global_step']
        self.input_size= client_params['input_size']
        self.output_size= client_params['output_size']


        adj = np.asarray(nx.adjacency_matrix(self.subGraph).todense())

        # print('adj shape', adj.shape)

        self.adj =torch.Tensor(adj + sp.eye(adj.shape[0])).to(device)

        self.device = device

        print('device: ', device)


        if self.args.use_gcn_m:
            self.MendGCN = MendGCN(self.args, self.subGraph, self.args.hidden_size, self.nodes_per_client, self.device).to(self.device)  # 补图模型 args.hidden_size

        self.init_base_model(None)

        self.train_dataloader =DataLoader(self.train_dataset, batch_size=self.batch_size) #DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size) #DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size) # DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


        self.global_step = start_global_step


        if self.args.adp_model:

            self.adp = F.sigmoid( nn.Parameter(torch.rand(len(self.base_model.state_dict())), requires_grad=True) )#.to(device))


    def forward(self, x, mendGcn_encoding):

        return self.base_model(x, self.global_step, mendGcn_encoding=mendGcn_encoding)



    def init_base_model(self, state_dict):
        self.base_model = GRUSeq2SeqWithGraphNet(self.input_size, self.output_size, self.base_model_kwargs).to(self.device)
        if state_dict is not None:
            self.base_model.load_state_dict(state_dict)
            self.base_model.to(self.device)

        if self.args.use_gcn_m:
            self.optimizer = getattr(torch.optim, self.optimizer_name)(
                itertools.chain(self.base_model.parameters(), self.MendGCN.parameters()), lr=self.lr,
                weight_decay=self.weight_decay)
        else:
            self.optimizer = getattr(torch.optim, self.optimizer_name)(
                self.base_model.parameters(), lr=self.lr,
                weight_decay=self.weight_decay)



    def local_train(self, state_dict_to_load, state_dict_gcn_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
            self.base_model.to(self.device)  # 初始化client模型
        if self.args.use_gcn_m:
            if state_dict_gcn_load is not None:
                self.MendGCN.load_state_dict(state_dict_gcn_load)
            self.MendGCN.to(self.device)
        self.train()
        with torch.enable_grad():
            for epoch_i in range(self.sync_every_n_epoch):  # 5， 每个client运行的的epoch

                print('client: {}'.format(self.client_i),
                'epoch : {}'.format(epoch_i ))

                num_samples = 0
                epoch_log = defaultdict(lambda : 0.0)



                self.train_dataloader.shuffle()
                for iter, (x, y) in enumerate(self.train_dataloader.get_iterator()):

                    x =torch.Tensor(x).to(self.device)  # B, T, N, F
                    y =torch.Tensor(y).to(self.device)

                    data = dict(
                        x=x,  y=y,   # B, T, N, F
                        adj = self.adj
                    )




                    if self.args.use_gcn_m:

                        x_input = x.permute(1, 0, 2, 3).flatten(1, 2)  # T x (B x N) x F

                        _, h_encode = self.base_model.encoder(x_input)  # layers x (B x N) x hidden_dim

                        graph_encoding = h_encode.view(h_encode.shape[0], x.shape[0], x.shape[2],
                                                       h_encode.shape[2]).permute(2, 1, 0, 3)  # N, B，layers， F

                       # print('shape ', graph_encoding.shape)

                        mendGraph_encoding = self.MendGCN(
                            graph_encoding)

                        mendGcn_encoding = mendGraph_encoding.permute(2, 1, 0, 3).flatten(1, 2)  # layers, B*N, hidden_dim
                    else:
                        mendGcn_encoding=[]
                    #    print('encode shape none')


                    y_pred = self(data, mendGcn_encoding)  # 预测结果  # B, T, N,1

                    loss = utils.masked_mae(y_pred, y, 0)

                   # print('loss', loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    num_samples += x.shape[0]

                    metrics = unscaled_metrics(y_pred, y, self.feature_scaler, 'train') # mse, mae, mape

                    epoch_log['train/loss'] += loss.detach() * x.shape[0]
                    for k in metrics:
                        epoch_log[k] += metrics[k] * x.shape[0]
                    self.global_step += 1

                    if iter % self.args.print_every == 0:
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train RMSE: {:.4f}' #Train MAPE: {:.4f},
                        print(log.format(iter, loss, metrics['train/mae'],  metrics['train/rmse']), flush=True)  #metrics['train/mape'],

                for k in epoch_log:
                    epoch_log[k] /= num_samples
                    epoch_log[k] = epoch_log[k].cpu()



        print(
              'client: {}'.format(self.client_i),
              'loss_train: {:.4f}'.format(epoch_log['train/loss'] ),
              'mae: {:.4f}'.format(epoch_log['train/mae'] ),
              'rmse: {:.4f}'.format(epoch_log['train/rmse']),
              )

        # self.cpu()
        state_dict = self.base_model.to('cpu').state_dict()
        if self.args.use_gcn_m:
            if self.args.avg_gcn:
                state_dict_gcn= self.MendGCN.to('cpu').state_dict()
        epoch_log['num_samples'] = num_samples
        epoch_log['global_step'] = self.global_step
        epoch_log = dict(**epoch_log)

        if self.args.use_gcn_m:
            if self.args.avg_gcn:
                return {
                    'state_dict': state_dict,
                    'state_dict_gcn': state_dict_gcn,
                    'log': epoch_log
                }
            else:
                return {
                    'state_dict': state_dict,
                    'log': epoch_log
                }
        else:
            return {
                'state_dict': state_dict,
                'log': epoch_log
            }



    def local_eval(self, dataloader, name, state_dict_to_load, state_dict_gcn_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
            self.base_model.to(self.device)
        if self.args.use_gcn_m:
            if state_dict_gcn_load is not None:
                self.MendGCN.load_state_dict(state_dict_gcn_load)
            self.MendGCN.to(self.device)


        self.eval()
        if name=='test' and self.args.save_r:
            y_true=[]
            y_pre=[]


        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda : 0.0)
            print('client: {}'.format(self.client_i),
                  name + ' ..... ')



            for iter, (x, y) in enumerate(dataloader.get_iterator()):


                x = torch.Tensor(x).to(self.device)  # B, T, N, F
                y = torch.Tensor(y).to(self.device)

                data = dict(
                    x=x, y=y,  # B, T, N, F
                    adj = self.adj
                )

                if self.args.use_gcn_m:

                    x_input = x.permute(1, 0, 2, 3).flatten(1, 2)  # T x (B x N) x F

                    _, h_encode = self.base_model.encoder(x_input)  # layers x (B x N) x hidden_dim

                    graph_encoding = h_encode.view(h_encode.shape[0], x.shape[0], x.shape[2],
                                                   h_encode.shape[2]).permute(2, 1, 0, 3)  # N, B，layers， F

                    mendGraph_encoding = self.MendGCN(
                        graph_encoding)

                    mendGcn_encoding = mendGraph_encoding.permute(2, 1, 0, 3).flatten(1, 2)  # layers, B*N, hidden_dim

                else:
                    mendGcn_encoding=[]

                y_pred= self(data,  mendGcn_encoding) #  # B, T, N,1

                loss = utils.masked_mae(y_pred, y, 0)

                num_samples += x.shape[0]

                metrics = unscaled_metrics(y_pred, y, self.feature_scaler, name)

                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]

                if name=='test' and self.args.save_r:
                    y = y.detach().cpu() * self.feature_scaler['_std'] + self.feature_scaler['_mean']
                    y_pred = y_pred.detach().cpu() * self.feature_scaler['_std'] + self.feature_scaler['_mean']
                    y_true.append(y)   # B, T, N, 1
                    y_pre.append(y_pred)

                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]

            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()

        if name=='test' and self.args.save_r:
            y_true = torch.cat(y_true, dim=0) # L, T, N,1
            y_pred = torch.cat(y_pre, dim=0)
            np.save('./{}_client{}_{}_true.npy'.format(self.args.dataset, self.args.num_clients,self.client_i), y_true.cpu().numpy())
            np.save('./{}_client{}_{}_pred.npy'.format(self.args.dataset,  self.args.num_clients,self.client_i), y_pred.cpu().numpy())



        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)

        return {'log': epoch_log}

    def local_tests(self, dataloader, name, state_dict_to_load, state_dict_gcn_load):
        if state_dict_to_load is not None:
            self.base_model.load_state_dict(state_dict_to_load)
            self.base_model.to(self.device)
        if self.args.use_gcn_m:
            if state_dict_gcn_load is not None:
               self.MendGCN.load_state_dict(state_dict_gcn_load)
            self.MendGCN.to(self.device)


        self.eval()
        if name=='test' and self.args.save_r:
            y_true=[]
            y_pre=[]


        with torch.no_grad():
            num_samples = 0
            epoch_log = defaultdict(lambda : 0.0)
            print('client: {}'.format(self.client_i),
                  name + ' ..... ')



            for iter, (x, y, real) in enumerate(dataloader.get_iterator()):


                x = torch.Tensor(x).to(self.device)  # B, T, N, F
                y = torch.Tensor(y).to(self.device)
                real = torch.Tensor(real).to(self.device)

                data = dict(
                    x=x, y=y,  # B, T, N, F
                    adj = self.adj

                )



                if self.args.use_gcn_m:

                    x_input = x.permute(1, 0, 2, 3).flatten(1, 2)  # T x (B x N) x F

                    _, h_encode = self.base_model.encoder(x_input)  # layers x (B x N) x hidden_dim

                    graph_encoding = h_encode.view(h_encode.shape[0], x.shape[0], x.shape[2],
                                                   h_encode.shape[2]).permute(2, 1, 0, 3)  # N, B，layers， F

                    mendGraph_encoding = self.MendGCN(
                        graph_encoding)

                    mendGcn_encoding = mendGraph_encoding.permute(2, 1, 0, 3).flatten(1, 2)  # layers, B*N, hidden_dim

                else:
                    mendGcn_encoding=[]

                y_pred= self(data,  mendGcn_encoding) #  # B, T, N,1


                loss = utils.masked_mae(y_pred, real, 0)

                num_samples += x.shape[0]

                metrics = unscaled_metrics(y_pred, real, self.feature_scaler, name, self.args.mape)

                epoch_log['{}/loss'.format(name)] += loss.detach() * x.shape[0]

                if name=='test' and self.args.save_r:
                    real = real.detach().cpu() * self.feature_scaler['_std'] + self.feature_scaler['_mean']
                    y_pred = y_pred.detach().cpu() * self.feature_scaler['_std'] + self.feature_scaler['_mean']
                    y_true.append(real)   # B, T, N, 1
                    y_pre.append(y_pred)

                for k in metrics:
                    epoch_log[k] += metrics[k] * x.shape[0]

            for k in epoch_log:
                epoch_log[k] /= num_samples
                epoch_log[k] = epoch_log[k].cpu()

        if name=='test' and self.args.save_r:
            y_true = torch.cat(y_true, dim=0) # L, T, N,1
            y_pred = torch.cat(y_pre, dim=0)
            np.save('./{}_{}_true.npy'.format(self.args.dataset, self.client_i), y_true.cpu().numpy())
            np.save('./{}_{}_pred.npy'.format(self.args.dataset, self.client_i), y_pred.cpu().numpy())

        epoch_log['num_samples'] = num_samples
        epoch_log = dict(**epoch_log)

        return {'log': epoch_log}

    def local_validation(self, state_dict_to_load, state_dict_gcn):
        return self.local_eval(self.val_dataloader, 'val', state_dict_to_load, state_dict_gcn)

    def local_test(self, state_dict_to_load, state_dict_gcn):
        return self.local_eval(self.test_dataloader, 'test', state_dict_to_load, state_dict_gcn)





