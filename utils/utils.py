import os
import pickle

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy.sparse as sp


def id_to_idloc(ids):
    idlocs= np.copy(ids)
    for i in range(len(ids)):
        idlocs[i]=ids.index(ids[i])

    return idlocs


def load_data(data_filename, num_client=2, num_of_hours=1, num_of_days=0, num_of_weeks=0, points_per_hour=6):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(data_filename).split('.')[0]

    dirpath = os.path.dirname(data_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_client' + str(num_client) +'_fedgcn' #+'_p'+ str(points_per_hour)


    print('load file:', filename)


    file_data = np.load(filename + '.npz', allow_pickle=True)['data']


    return file_data

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(dataset_name):

    if dataset_name == 'PEMS04':
        adj_mx_path = './data/sensor_graph/dis_adj_index_04.csv'
        adj_mx = pd.read_csv(adj_mx_path, header=None).values
    elif dataset_name == 'PEMS08':
        adj_mx_path = './data/sensor_graph/dis_adj_index_08.csv'
        adj_mx = pd.read_csv(adj_mx_path, header=None).values

    elif dataset_name == 'PEMS03':
        adj_mx_path = './data/sensor_graph/dis_adj_index_03.csv'
        adj_mx = pd.read_csv(adj_mx_path, header=None).values
    elif dataset_name == 'PEMS07':
        adj_mx_path = './data/sensor_graph/dis_adj_index_07.csv'
        adj_mx = pd.read_csv(adj_mx_path, header=None).values
    elif dataset_name == 'PEMS228':
        adj_mx_path = './data/sensor_graph/dis_W_index_228.csv'
        adj_mx = pd.read_csv(adj_mx_path, header=None).values

    elif dataset_name == 'PEMS-BAY':
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle('./data/sensor_graph/adj_mx_bay.pkl')
    else:
        raise ValueError

    return adj_mx

def normalize( mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def unscaled_metrics(y_pred, y, scaler, name, mape=False):
    # if name=='test' and mape==True:
    #     y=y.detach().cpu()
    # else:
    y = y.detach().cpu()*scaler['_std'] + scaler['_mean']
    y_pred = y_pred.detach().cpu()*scaler['_std'] + scaler['_mean']

    rmse = masked_rmse(y_pred, y, 0.0)
    mae = masked_mae(y_pred, y, 0.0)

    # MAPE
   # mape = torch.abs((y_pred - y) / y).mean()
    mape = masked_mape(y_pred, y, 0.0)
    return {
        #'{}/mse'.format(name): mse.detach(),  # rmse
        '{}/rmse'.format(name): rmse.detach(),
        '{}/mae'.format(name): mae.detach(),
        '{}/mape'.format(name): mape.detach()
    }

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan, eps=1e-02):
    if np.isnan(null_val):
      #  labels = torch.where(torch.abs(labels) < 1, torch.zeros_like(labels), labels)
       # mask = torch.abs(labels) > 1
        mask =  ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
       # mask = torch.abs(labels) > 1
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.abs(torch.where(torch.isnan(mask), torch.zeros_like(mask), mask))
    loss = torch.abs(preds-labels)/(labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.abs(torch.mean(loss))




def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse

class DataLoader(object):
    def __init__(self, data, batch_size, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0

        self.nodes=data[0].shape[-2]

        xs=data[0]
        ys=data[1]
        if pad_with_last_sample:   #当数据长度不是batch_size的整数时，用最后一个数填充不足的部分
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]    #获取一个[batchsize,12,num_nodes,input_dims]的数据
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()