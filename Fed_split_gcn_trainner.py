import math

from copy import deepcopy
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

import utils
from FedGCN_train import train_mendgcn
from MendGCN import MendGCN
from dataowner import DataOwner
from utils.utils import load_adj, DataLoader

#from models import GRUSeq2SeqWithGraphNet
from models.GCNSeq2SeqM import GRUSeq2SeqWithGraphNet
from utils.utils import unscaled_metrics

#from utils import config
import networkx as nx
from utils import utils

from utils.utils import load_data
import itertools
from Fed_gcn_client import FedNodePredictorClient



class SplitFedNodePredictor( ):
    def __init__(self, hparams, device):
        super().__init__()
        self.hparams = hparams
        self.base_model = None
        self.gcn = None

        self.device = device

        self.setup()


    def setup(self):
        # must avoid repeated init!!! otherwise loaded model weights will be re-initialized!!!
        if self.base_model is not None:
            return
        data = load_data(self.hparams.datafile,self.hparams.num_clients )  # 加载数据集
        self.data = data
        #scale=data['stats']

        # Each node (client) has its own model and optimizer
        # Assigning data, model and optimizer for each client
        num_clients = self.hparams.num_clients
        num_nodes = data[0]['train']['x'].shape[2]   # B, T, N, F
        nodes_per_client = num_nodes # 数据已经进行了划分

        input_size = self.data[0]['train']['x'].shape[-1]
        output_size = self.data[0]['train']['target'].shape[-1]  # 预测步长？ B,T, N, 1


        print('generate global graph......')
        adj_mx = load_adj(dataset_name=self.hparams.dataset) # 读取0-1邻接矩阵

       # adj = sp.coo_matrix(adj_mx)
        # edge_index = np.vstack((adj.row, adj.col)).transpose() # E, 2


        print('preprocessing data.........')

        subGraphs= DataOwner(adj_mx, num_clients, nodes_per_client)


        client_params_list = []

        self.data=[]

        for client_i in range(num_clients):
            client_datasets = {}

            for name in ['train', 'val', 'test']:
                client_datasets[name] = (
                    torch.from_numpy(data[client_i][name]['x']).float(),
                    torch.from_numpy(data[client_i][name]['target']).float()
                )



            self.data.append(torch.from_numpy(data[client_i]['train']['x']).float())


            print('client model parameters.........')

            client_params = {}
            client_params.update(  # 字典合并，相同的键就覆盖
                train_dataset=client_datasets['train'],
                val_dataset=client_datasets['val'],
                test_dataset=client_datasets['test'],
                feature_scaler=data[client_i]['stats'],
                input_size=input_size,
                output_size=output_size,
                start_global_step=0,
                subGraphs=subGraphs.subGs[client_i],
                client_i=client_i,
                nodes_per_client= nodes_per_client,
                args = self.hparams
            )
            client_params_list.append(client_params) # 每个用户的数据和参数设置


        self.client_params_list = client_params_list # 所有client的参数结构列表

        # 定义global 模型

        self.base_model = GRUSeq2SeqWithGraphNet(input_size= input_size, output_size=output_size,
                                                                              args = self.hparams)  # global model

        self.client_models=[]
        for client_i, client_params in enumerate(self.client_params_list):  #定义client model

            print(str(client_i) +'  client model define....')

            client_model  = FedNodePredictorClient(client_params, self.device)
            self.client_models.append(client_model)

        return self.data


    def training_step(self):
        # 1. train locally and collect uploaded local train results
        local_train_results = []

        print('clients training....')

        state_dict = self.base_model.to('cpu').state_dict()

        state_dict_gcn=None

        if self.hparams.use_gcn_m:
            if self.hparams.avg_gcn and self.gcn is not None:
                state_dict_gcn = self.gcn


        for client_i in range(self.hparams.num_clients):  #每个local模型遍历训练

            print(str(client_i) +'  training....')

            local_train_result = self.client_models[client_i].local_train(state_dict, state_dict_gcn)
            local_train_results.append(local_train_result)  # 训练结果，字典， state_dict: , log:

        # update global steps for all clients
        for ltr, client_params in zip(local_train_results, self.client_params_list):
            client_params.update(start_global_step=ltr['log']['global_step'])

        # 2. aggregate (optional? kept here to save memory, otherwise need to store 1 model for each node)
        agg_local_train_results = self.aggregate_local_train_results(local_train_results) # 对log保存的结果和client模型求平均

       # self.MGCN_models = [ltr['state_dict_gcn'] for ltr in local_train_results]

        # 2.1. update aggregated weights
        if agg_local_train_results['state_dict'] is not None:
            self.base_model.load_state_dict(agg_local_train_results['state_dict'])  # 更新FedAVG更新后的模型


        if self.hparams.avg_gcn:
            if agg_local_train_results['state_dict_gcn'] is not None:
                self.gcn = agg_local_train_results['state_dict_gcn']  # 状态字典

        agg_log = agg_local_train_results['log']
        log = agg_log

        print(' client model traing epochs end')
        return {'loss': torch.tensor(0).float(), 'progress_bar': log, 'log': log}

    def aggregate_local_train_results(self, local_train_results):
        if self.hparams.use_gcn_m:
            if self.hparams.avg_gcn:
                return {
                    'state_dict': self.aggregate_local_train_state_dicts(
                        [ltr['state_dict'] for ltr in local_train_results]  # 模型求平均
                    ),
                    'state_dict_gcn': self.aggregate_local_train_state_dicts(
                        [ltr['state_dict_gcn'] for ltr in local_train_results]),  # MendGCN不做平均处理
                    'log': self.aggregate_local_logs(
                        [ltr['log'] for ltr in local_train_results]
                    )
                }
            else:
                return {
                    'state_dict': self.aggregate_local_train_state_dicts(
                        [ltr['state_dict'] for ltr in local_train_results]  # 模型求平均
                    ),
                    'log': self.aggregate_local_logs(
                        [ltr['log'] for ltr in local_train_results]
                    )
                }

        else:
            return {
                'state_dict': self.aggregate_local_train_state_dicts(
                    [ltr['state_dict'] for ltr in local_train_results]  # 模型求平均
                ),

                'log': self.aggregate_local_logs(
                    [ltr['log'] for ltr in local_train_results]
                )
            }



    def aggregate_local_train_state_dicts(self, local_train_state_dicts):  # 状态字典列表
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:  # 状态字典中的参数结构的键
            agg_state_dict[k] = 0  # 参数值初始化为0
            for ltsd in local_train_state_dicts:
                agg_state_dict[k] += ltsd[k]  # 值相加
            agg_state_dict[k] /= len(local_train_state_dicts)  # 求平均， 包括mendgcn

        return agg_state_dict

    def aggregate_local_logs(self, local_logs, selected=None):
        agg_log = deepcopy(local_logs[0])  # epoch_log 字典

        for k in agg_log: # log 中键
            agg_log[k] = 0

            for local_log_idx, local_log in enumerate(local_logs): # 遍历epoch_log字典列表，local_log_idx 为下标，local_log为epoch_log字典
                if k == 'num_samples':
                    agg_log[k] += local_log[k] # 字典中的键 K对应的值相加
                else:
                    agg_log[k] += local_log[k] * local_log['num_samples']

        for k in agg_log:
            if k != 'num_samples':
                agg_log[k] /= agg_log['num_samples']

        return agg_log

    def training_epoch_end(self, outputs):
        # already averaged!
        log = outputs[0]['log']
        log.pop('num_samples')
        return {'log': log, 'progress_bar': log}

    def validation_step(self):

        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        state_dict = self.base_model.to('cpu').state_dict()

        state_dict_gcn = None

        if self.hparams.use_gcn_m:
            if self.hparams.avg_gcn and self.gcn is not None:
                state_dict_gcn = self.gcn

        for client_i in range(self.hparams.num_clients):  # 每个local模型遍历验证

            print(str(client_i) + '  validating....')

            local_val_result = self.client_models[client_i].local_validation(state_dict, state_dict_gcn)
            local_val_results.append(local_val_result)  # 验证结果，字典， state_dict: , log:

        # 2. aggregate
        log = self.aggregate_local_logs([x['log'] for x in local_val_results]) # log取平均之后的值

        return {'progress_bar': log, 'log': log}

    def validation_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)

    def test_step(self):

        # 1. vaidate locally and collect uploaded local train results
        local_val_results = []
        state_dict = self.base_model.to('cpu').state_dict()

        state_dict_gcn = None

        if self.hparams.use_gcn_m:
            if self.hparams.avg_gcn and self.gcn is not None:
                state_dict_gcn = self.gcn


        for client_i in range(self.hparams.num_clients):
            local_val_result = self.client_models[client_i].local_test( state_dict, state_dict_gcn)
            local_val_results.append(local_val_result) # 测试结果 log字典


        # 2. aggregate
        # separate seen and unseen nodes if necessary
        log = self.aggregate_local_logs([x['log'] for x in local_val_results]) # log取平均后的值

        return {'progress_bar': log, 'log': log}

    def test_epoch_end(self, outputs):
        return self.training_epoch_end(outputs)


