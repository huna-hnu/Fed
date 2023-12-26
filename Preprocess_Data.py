import math
import os
import pickle

import numpy as np
import argparse
import configparser
import pandas as pd



def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + points_per_hour   #num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS03':
        data_path = os.path.join('./data/PEMS03/pems03.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMS07':
        data_path = os.path.join('./data/PEMS07/pems07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMS04':
        data_path = os.path.join('./data/PEMS04/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        data_path = os.path.join('./data/PEMS08/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS-BAY':
        data_path = os.path.join('./data/PEMS-BAY/pems-bay.h5')
        data = pd.read_hdf(data_path)  # onley the first dimension, traffic flow data
    elif dataset == "METR-LA":
        data_path = os.path.join('./data/METR-LA/metr-la.h5')
        data = pd.read_hdf(data_path)  # onley the first dimension, traffic flow data
    elif dataset=='PEMS228':
        data_path = os.path.join('./data/PEMS228/V_228.csv')
        data = pd.read_csv(data_path, header=None).values  # onley the first dimension, traffic flow data

    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data



def read_and_generate_dataset(dataset_name,graph_signal_matrix_filename,
                                                     num_of_weeks, num_of_days,
                                                     num_of_hours, num_for_predict,
                                                     points_per_hour=12, num_client= 2, adj_filename='', save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    #data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)
    # data_seq=pd.read_csv(graph_signal_matrix_filename, header=None).values
    # data_seq = np.expand_dims(data_seq, 2)
    #data_seq = pd.read_hdf(graph_signal_matrix_filename).values
    data_seq = load_st_dataset(dataset_name)




    num_nodes=data_seq.shape[1]
    client_nodes= math.ceil(num_nodes/num_client) # 上界

    client_data=[]
    for i in range(num_client):

        data=data_seq[:,i*client_nodes:min((i+1)*client_nodes, num_nodes)]

       # adj= adj_mx[i*client_nodes:(i+1)*client_nodes,i*client_nodes:(i+1)*client_nodes]

        all_samples = []
        for idx in range(data.shape[0]):
            sample = get_sample_indices(data, num_of_weeks, num_of_days,
                                        num_of_hours, idx, num_for_predict,
                                        points_per_hour)
            if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
                continue

            week_sample, day_sample, hour_sample, target = sample

            sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

            if num_of_weeks > 0:
                week_sample = np.expand_dims(week_sample, axis=0)# 1, T, N, F .transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(week_sample)

            if num_of_days > 0:
                day_sample = np.expand_dims(day_sample, axis=0)# 1, T, N, F .transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(day_sample)

            if num_of_hours > 0:
                hour_sample = np.expand_dims(hour_sample, axis=0)# # 1, T, N, F.transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0)[:, :, :, 0:1] ## 1, T, N, 1.transpose((0, 2, 3, 1))[:, :, 0:1, :]  # (1,N,1,T)
            sample.append(target)

            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)

            all_samples.append(
                sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

        split_line1 = int(len(all_samples) * 0.6)
        split_line2 = int(len(all_samples) * 0.8)

        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]  # [(B,Tw,N,F),(B,Td,N,F),(B,Th,N,F),(B,Tpre,N,1),(B,1)]
        validation_set = [np.concatenate(i, axis=0)
                          for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0)
                       for i in zip(*all_samples[split_line2:])]

        train_x = np.concatenate(training_set[:-2], axis=1)  # (B,T',N,F) T'=Th+Td+Tw
        val_x = np.concatenate(validation_set[:-2], axis=1)
        test_x = np.concatenate(testing_set[:-2], axis=1)

        train_target = training_set[-2]  # (B,T, N,1)
        val_target = validation_set[-2]
        test_target = testing_set[-2]

        train_timestamp = training_set[-1]  # (B,1)
        val_timestamp = validation_set[-1]
        test_timestamp = testing_set[-1]

        (stats, train_x_norm, val_x_norm, test_x_norm, train_target_norm, val_target_norm, test_target_norm) = normalization(train_x, val_x, test_x, train_target, val_target, test_target)

        all_data = {
            'train': {
                'x': train_x_norm,
                'target': train_target_norm,
                'timestamp': train_timestamp,
            },
            'val': {
                'x': val_x_norm,
                'target': val_target_norm,
                'timestamp': val_timestamp,
            },
            'test': {
                'x': test_x_norm,
                'target': test_target_norm,
                 'real':test_target,
                'timestamp': test_timestamp,
            },
            'stats': {
                '_mean': stats['_mean'],
                '_std': stats['_std'],
            }
            # 'adj':{'adj':adj,
            # }
        }
        print('train x:', all_data['train']['x'].shape)
        print('train target:', all_data['train']['target'].shape)
        print('train timestamp:', all_data['train']['timestamp'].shape)
        print()
        print('val x:', all_data['val']['x'].shape)
        print('val target:', all_data['val']['target'].shape)
        print('val timestamp:', all_data['val']['timestamp'].shape)
        print()
        print('test x:', all_data['test']['x'].shape)
        print('test target:', all_data['test']['target'].shape)
        print('test timestamp:', all_data['test']['timestamp'].shape)
        print()
        print('train data _mean :', stats['_mean'].shape, stats['_mean'])
        print('train data _std :', stats['_std'].shape, stats['_std'])

        client_data.append(all_data)

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(
            num_of_weeks)) + '_client' + str(num_client) + '_fedgcn'
        print('save file:', filename)
        np.savez_compressed(filename,
                            data=client_data
                            )
        print('end')
    return client_data


def normalization(train, val, test, train_y, val_y, test_y):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T) -> B, T, N, F
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)  # default axis=(0,1,3)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    train_y = normalize(train_y)
    val_y = normalize(val_y)
    test_y = normalize(test_y)


    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm, train_y, val_y, test_y


# prepare dataset
# parser = argparse.ArgumentParser()
# parser.add_argument("--config", default='Configurations/PEMS03_astgcn.conf', type=str,
#                     help="configuration file path")
# args = parser.parse_args()
# config = configparser.ConfigParser()
# print('Read configuration file: %s' % (args.config))
# config.read(args.config)
# data_config = config['Data']
# training_config = config['Training']
#
# adj_filename = data_config['adj_filename']
# graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
# if config.has_option('Data', 'id_filename'):
#     id_filename = data_config['id_filename']
# else:
#     id_filename = None

num_of_vertices = 207#int(data_config['num_of_vertices'])
points_per_hour =12# int(data_config['points_per_hour'])
num_for_predict = 12#int(data_config['num_for_predict'])
len_input = 12#int(data_config['len_input'])
dataset_name ='METR-LA'# data_config['dataset_name']
num_of_weeks =0# int(training_config['num_of_weeks'])
num_of_days =0# int(training_config['num_of_days'])
num_of_hours =1# int(training_config['num_of_hours'])
graph_signal_matrix_filename = './data/METR-LA/metr-la.h5'#data_config['graph_signal_matrix_filename']

adj_file= ''#data_config['adj_filename']
num_client =6

all_data = read_and_generate_dataset(dataset_name,graph_signal_matrix_filename, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour, num_client, adj_file, save=True)
