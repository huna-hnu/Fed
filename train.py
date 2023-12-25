import os
import time
from argparse import ArgumentParser

import numpy as np
import torch

import models
from Fed_split_gcn_trainner import SplitFedNodePredictor

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS04') # for read adj
parser.add_argument('--datafile', type=str, default='data/PEMS04/pems04.npz')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--device', type=str, default="cuda:0") #

parser.add_argument('--save',type=str,default='./artifacts/pems08',help='save path')   # default: metr


    # model parameters
parser.add_argument('--hidden_size', type=int, default=64)  # for gru
parser.add_argument('--dropout_t', type=float, default=0)  #
#parser.add_argument('--cl_decay_steps', type=int, default=1000)
parser.add_argument('--use_curriculum_learning', action='store_true')
parser.add_argument('--gru_num_layers', type=int, default=1)


    # gcn layers 图卷积网络的层数
#parser.add_argument('--gn_layer_num', type=int, default=1)
parser.add_argument('--hidden', type=int, default=32) # hidden dim for gcn
parser.add_argument('--dropout', type=float, default=0.1) # gcn drop
parser.add_argument('--dropout_m', type=float, default=0.1)  #gen drop
parser.add_argument('--latent_dim', type=int, default=128)

parser.add_argument('--num_pred', type=int, default=4)

parser.add_argument('--num_clients', type=int, default=6)


parser.add_argument('--save_r', type = bool, default=False)  # 预测值和真实值保存

parser.add_argument('--use_gcn', type = bool, default=False)  # 是否使用MendGCN，encoder之前
parser.add_argument('--use_gcn_m', type = bool, default=True)  # 在中间使用MendGCN
parser.add_argument('--use_gcn_t', type = bool, default=False)  # GRUseq2Seq是否使用GCN
parser.add_argument('--avg_gcn', type = bool, default=False)  # mendgcn模型求平均
parser.add_argument('--use_fed_g', type = bool, default=False)  # train_mendgcn调用
   # 训练参数
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sync_every_n_epoch', type=int, default=1) #client model训练迭代轮次
parser.add_argument('--Optimizer_name', type=str, default="Adam")
parser.add_argument('--print_every', type=int, default=50)

parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--cuda', type = bool, default=True)


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
print('drop gcn', args.dropout)
print('num predict', args.num_pred)
#print('hidden rate', args.hidden_rate)
torch.cuda.empty_cache()
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark=False



engine = SplitFedNodePredictor(args, device)  # 初始化训练器

print("start training...", flush=True)
his_loss = []
val_time = []
train_time = []
test_time = []
    #  rate = 0.9
for i in range(1, args.epochs + 1):

        # 训练
    t1 = time.time()
    results = engine.training_step()

    #model= results['state_dict']

    log= results['log']

    loss=log['train/loss']

    mae= log['train/mae']
    rmse=log['train/rmse']
  #  mape= log['train/mape']


    log = 'epoch: {}, Train mae: {:.4f}, Train RMSE: {:.4f}' # Train MAPE: {:.4f},
    print(log.format(i, mae, rmse), flush=True)  #mape,

    t2 = time.time()
    train_time.append(t2 - t1)



       # 模型评估
    s1 = time.time()

    val_results = engine.validation_step()

    s2 = time.time()
    log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
    print(log.format(i, (s2 - s1)))
    val_time.append(s2 - s1)

    eval_log = val_results['log']
    for k in eval_log:
        print(k, eval_log[k])

    his_loss.append(eval_log['val/mae'])

    loss = eval_log['val/mae'].numpy()


    log = 'Epoch: {}, Valid Loss: {:.4f},  Valid mae: {:.4f}, Valid_RMSE: {:.4f},  Training Time: {:.4f}/epoch' #Valid MAPE: {:.4f},
    print(log.format(i, eval_log['val/loss'], eval_log['val/mae'], eval_log['val/rmse'],  (t2 - t1)), #eval_log['val/mape'],
              flush=True)
    torch.save(engine.base_model.state_dict(), args.save + "_epoch_" + str(i) + "_" + str(
            np.round( loss, 2)) + ".pth")  # state_dict变量存放训练过程中需要学习的权重和偏执系数，

print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    # 测试模型
tt=time.time()
bestid = np.argmin(his_loss)  # 选择效果最好的epoch  来对测试集进行测试
print("The valid loss on best model is", str(bestid), str(np.round(his_loss[bestid].numpy(), 4)))
engine.base_model.load_state_dict(
    torch.load(args.save + "_epoch_" + str(bestid + 1) + "_" + str(np.round(his_loss[bestid].numpy(), 2)) + ".pth"))

test_results = engine.test_step()  # 模型测试

test_log = test_results['log']

print('drop gcn', args.dropout)
print('num predict', args.num_pred)

log = ' Test Loss: {:.4f},  Test MAE: {:.4f},  Test RMSE: {:.4f},  Test Time: {:.4f}/epoch'  #Test MAPE: {:.4f},
print(
    log.format( test_log['test/loss'], test_log['test/mae'], test_log['test/rmse'], (time.time()-tt)), #, test_log['test/mape'
        flush=True)

