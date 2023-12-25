import MendGCN
import torch
import torch.nn.functional as F
from torch import optim

from models import feat_loss

import numpy as np
import time

def train_mendgcn(args, model, MendGCNs ,data): # L, T, N, F
    feat_shape=args.hidden_size

    local_gen_list=[]
    optim_list=[]
    inputf=[]
    inputf_all=[]
    t=time.time()
    model.requires_grad_(False)
    for i in range(args.num_clients):
        MendGCNs[i].set_fed_model()
        local_gen_list.append(MendGCNs[i].fed_model.gen)

        local_gen_list.append(MendGCNs[i].neighgen)
        optim_list.append(optim.Adam(local_gen_list[-1].parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay))

        #print('shape', data[i].shape) # L, T, N, F

        input_feat = torch.mean(data[i], dim=1)  # L, N, F
        _, input_feat = model.encoder(input_feat)  # layers, N, F
        input_feat = input_feat.permute(1, 0, 2).unsqueeze(1).to(args.device)  # N, 1, layers, F
        inputf_all.append(input_feat)
        MendGCNs[i].get_train_test_feat_targets(input_feat)

        input_feat= MendGCNs[i].all_feat

       # print('shape', input_feat.shape)
        inputf.append(input_feat)

    for epoch in range(args.gen_epochs):  # 修补图模型训练
        for i in range(args.num_clients):


            local_gen_list[i].train()
            optim_list[i].zero_grad()
            local_model = MendGCNs[i].neighgen
            input_feat= inputf[i]  # N, 1, layers, F
            input_edge = MendGCNs[i].edges
            input_adj = MendGCNs[i].adj
            output_missing, output_feat, outgcn_feat = local_model(input_feat, input_edge, input_adj)   # #1, layers, N+N*num_pred, F
            output_missing = torch.flatten(output_missing)
            output_feat = output_feat.view(
            output_feat.shape[0], output_feat.shape[1], len(MendGCNs[i].all_ids), args.num_pred,  MendGCNs[i].feat_shape) # # 1, layers, N, num_pred, F

            loss_train_missing = F.smooth_l1_loss(output_missing[MendGCNs[i].
                                                  train_ids].float(),
                                                  MendGCNs[i].all_targets_missing[MendGCNs[i].
                                                  train_ids].reshape(-1).float())

            loss_train_feat = feat_loss.greedy_loss(args, output_feat[:,:,MendGCNs[i].train_ids,:,:],
                                                    MendGCNs[i].all_targets_feat[:,:,MendGCNs[i].train_ids,:,:],
                                                    output_missing[MendGCNs[i].train_ids],
                                                    MendGCNs[i].all_targets_missing[
                                                        MendGCNs[i].train_ids
                                                    ]).unsqueeze(0).mean().float()




            for j in range(args.num_clients):
                if j != i:
                    choice = np.random.choice(len(list(MendGCNs[j].subG.nodes())),
                                              len(MendGCNs[i].train_ids), replace=False) # 子图中随机选择节点数
                    others_ids= np.array(MendGCNs[j].subG.nodes())[choice] #子图中部分节点
                    global_target_feat = []
                    for c_i in others_ids: # 遍历选中的节点
                        # neighbors_ids=np.array(MendGCNs[j].subG.__getitem__(c_i))  # c_i的邻居节点
                        flag=True
                        id_i = c_i
                        while flag == True: # 选取到邻居节点个数不为0的节点
                           # print('while')
                            #neighbors_ids = np.array(MendGCNs[j].subG.__getitem__(id_i)) # id_i的邻居节点
                            if len(np.array(MendGCNs[j].subG.__getitem__(id_i)))>0:
                                neighbors_ids = MendGCNs[j].subG.__getitem__(id_i)
                                flag = False
                            else:
                                c_i = np.random.choice(len(list(MendGCNs[j].subG.nodes())), 1)  # 随机选择一个值
                                id_i = list(MendGCNs[j].subG.nodes()).index(c_i)

                                  # MendGCNs[j].subG.nodes()[c_i] # 子图节点

                       # print(len(neighbors_ids), list(neighbors_ids))
                        choice_i = np.random.choice(list(neighbors_ids), args.num_pred) #在邻居节点中随机选num_pred个值
                        for ch_i in choice_i: # 遍历选中的邻居节点，将其特征收集
                            global_target_feat.append(inputf_all[j][ch_i,...]) #将选中的邻居节点特征收集 num_pred 1, 1, layers, F

                    global_target_feat = torch.cat(global_target_feat, dim=0).reshape(
                        (-1, inputf[0].shape[-2], len(MendGCNs[i].train_ids), args.num_pred,  feat_shape))  # L, 1, N, num_pred, F

                    # global_target_feat = global_target_feat.view(
                    #     len(MendGCNs[i].train_ids), args.num_pred, feat_shape)

                    loss_train_feat_other = feat_loss.greedy_loss(args, output_feat[:,:,MendGCNs[i].train_ids, :, :], # local预测的节点特征
                                                             global_target_feat, # 其他子图节点特征
                                                             output_missing[MendGCNs[i].train_ids],
                                                             MendGCNs[i].all_targets_missing[
                                                                  MendGCNs[i].train_ids]
                                                             ).unsqueeze(0).mean().float()
                    loss += args.c* loss_train_feat_other
            loss = 1.0 / args.num_clients * loss
            loss.requires_grad_(True)
            loss.backward()
            optim_list[i].step()

    for i in range(args.num_clients):
        MendGCNs[i].neighgen.load_state_dict(MendGCNs[i].fed_model.cpu().state_dict())
        MendGCNs[i].neighgen.to(args.device)

    return



