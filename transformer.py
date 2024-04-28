import graphs
import random
import copy
import time
import math
import numpy as np
import networkx as nx
# from scipy.spatial.distance import pdist, squareform, mahalanobis
# from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import Linear
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from create_dataset import MyDataset
# import node_features as  nf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from GCN import GCN2
from infltransformer_encoder import InflTransformerEncoder




def PredictNodeByDenseNet(x, edge_index, y, is_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GNN = GCN2(3, 64, 128).to(device)
    embedding = GNN(x, edge_index)
    # 扩充维度
    embedding = embedding.unsqueeze(0)

    encoder = InflTransformerEncoder(128, 64, 8, 64, 4, 0.1, 0.1, True, 0.1, 0.1).to(device)

    # optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    # loss_fun
    criterion = nn.MSELoss()                    # MSE均方误差损失函数   与L2loss有区别
    # criterion = nn.L1Loss()                    #MAE平均绝对误差损失函数
    # criterion = nn.SmoothL1Loss()               #Smooth L1损失函数

    # training loop
    if is_train:
        print("------------train start---------------")
        for epoch in range(500):
            encoder.train()  # 开启模型的训练模式，由于数据集较小，没有分batch训练（直接作为1个batch）
            optimizer.zero_grad()
            output = encoder(embedding)
            loss = criterion(output, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            state = {'net': encoder.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch + 1}
            torch.save(state, './data/atparameters/' + "111" + '.pth')
        print("------------train end---------------")
    else:
        checkpoint = torch.load('./data/atparameters/' + "111" + '.pth')
        encoder.load_state_dict(checkpoint['net'])
        y_pred = encoder(embedding)
        y_pred_sort = torch.argsort(y_pred.squeeze(), descending=True)   #返回对应的input(node)
        node_features_gcn = y_pred_sort.detach().numpy()
        return  node_features_gcn

if __name__ == '__main__':
    is_train = False
    if is_train:
        # 训练模型
        train_dataset = MyDataset("lesmis_77")
        x = torch.from_numpy(train_dataset.centrality_feature).type(torch.float)
        edge_index = torch.from_numpy(train_dataset.edge_index).type(torch.long)
        y = torch.from_numpy(train_dataset.labels).type(torch.float)
        #开始训练模型
        PredictNodeByDenseNet(x, edge_index, y, is_train)
    else:
        test_dataset = MyDataset("polbooks_105")
        xll = torch.from_numpy(test_dataset.centrality_feature).type(torch.float)
        edge_indexll = torch.from_numpy(test_dataset.edge_index).type(torch.long)
        yll = None
        candidate_seed = PredictNodeByDenseNet(xll, edge_indexll, yll, False)
        print(candidate_seed)

