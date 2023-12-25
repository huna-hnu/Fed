import networkx as nx
import numpy as np
import pandas as pd
import os
# import stellargraph as sg
import torch

import scipy.sparse as sp

MIN=1e-10
class DataOwner:
    def __init__(self,adj,client_num,subG_nodes): # 每个子图的节点数

        self.subGs=[]
        self.subDs=[] # 子图度矩阵
        self.subDMs =[]  # 度矩阵变化，丢失边数
        self.adjs =[]
        dd= adj.sum(1)
        D= np.diag(dd)

        for i in range(client_num):
            print('subG', i , len(adj))
            start = i*subG_nodes
            end= min((i+1)*subG_nodes, len(adj))

            subadj = adj[start:end,start:end]

            subD= np.diag(subadj.sum(1))

            subDM= D[start:end,start:end]-subD

            adj_index = sp.coo_matrix(subadj)

            subG = nx.Graph()

            subG.add_nodes_from(range(0, end-start))

            for j in range(len(adj_index.row)):
                subG.add_edge(adj_index.row[j], adj_index.col[j])

            print(list(subG.nodes()))

            self.subGs.append(subG)
            self.subDs.append(subD)
            self.subDMs.append(subDM)

            file='clientnum_'+str(client_num)+'client_'+str(i)+'_Degree_Matrix_Diff.npz'
            np.save(file, subDM)
















