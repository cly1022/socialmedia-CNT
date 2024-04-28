import os
import torch
import numpy as np
import glob
from node_features import Create_Centrality
from torch_geometric.data import Dataset, Data


class MyDataset(torch.utils.data.Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self, graph_name):
        #对继承自父类的属性进行初始化(好像没有这句也可以？？)
        super(MyDataset,self).__init__()
        # TODO
        #1、初始化一些参数和函数，方便在__getitem__函数中调用。
        #2、制作__getitem__函数所要用到的图片和对应标签的list。
        #也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        print("----start load dataset----")
        self.nodes_num = int(graph_name.split("_")[1])
        self.edge_index = np.load("data/features/" + graph_name + "_ei.npy")
        self.centrality_feature = np.load("data/features/" + graph_name + "_c3.npy", allow_pickle=True)
        self.labels = np.load("data/labels/" + graph_name + ".npy")
        print(len(self.centrality_feature))
        assert self.nodes_num == len(self.centrality_feature), ("中心性节点的维度有误")
        assert self.nodes_num == len(self.labels), ("labels的维度有误")
        print("load edge_index from:"  + "data/features/" + graph_name + "_ei.npy")
        print("load centrality_feature from:"  + "data/features/" + graph_name + "_c3.npy")
        print("load labels from:"  + "dataset/labels/" + graph_name + ".npy")
        print("----end load dataset----")
        pass
    def __getitem__(self, idx):
        # TODO
        #1、根据list从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        #2、预处理数据（例如torchvision.Transform）。
        #3、返回数据对（例如图像和标签）。
        #这里需要注意的是，这步所处理的是index所对应的一个样本。
        edge_index = self.edge_index[idx]
        x = self.centrality_feature[idx]
        y = self.labels[idx]
        """Convert ndarrays to Tensors."""
        # ELN 返回数据 edge_index, centrality_feature, lp_feature, nodewalkf_feature, node2vec_feature
        return torch.from_numpy(self.edge_index).long(), torch.from_numpy(x).float(), torch.from_numpy(y).float()
        
        pass
    def __len__(self):
        #返回数据集大小
        return self.nodes_num



# ELN 代码测试
# mydataset = MyDataset("Test_100_10")
# print(mydataset.edge_index)
# print(mydataset.edge_index.shape)
# a = 0

          





