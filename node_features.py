import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.io import mmread

# TO 1.中心性特征矩阵
def Create_DC(G):
    dc = list()
    # ELN 这里的邻接矩阵因为生成的图边是顺序的所以生成的邻接矩阵理论上也是顺序的,对节点进行限制就行
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix = np.array(adj_matrix, dtype=np.float32).reshape(len(adj_matrix), len(adj_matrix))
    dc = np.array(adj_matrix.sum(axis=0))
    # print(dc)

    # ELN 归一化操作
    norm = dc.max()
    dc= dc / norm
    return dc


def Hindex(indexList):
    indexSet = sorted(list(set(indexList)),reverse = True)
    for index in indexSet:
        #clist为大于等于指定引用次index的文章列表
        clist = [i for i in indexList if i >=index ]
        #由于引/用次数index逆序排列，当index<=文章数量/en(clist)时，得到H指数
        if index <=len(clist):
            break
        if index == indexSet[-1]:
            index = -1
    return index + 1

def get_neigbors(G, node, depth=1):
    output = {}
    layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

def Create_HI(G):
    # degrees = sorted([(node, G.degree(node)) for node in G.nodes()], key=lambda x:x[1], reverse=True)
    n = len(G.nodes())
    HI = list()
    for i in range(n):
        index =Hindex(get_neigbors(G, i)[1])
        HI.append(index)
    # print(HI)

    # ELN 归一化操作
    norm = np.array(HI).max()
    HI = HI / norm
    return HI

def k_shell(graph):
    importance_dict={}
    level=1
    while len(graph.degree):
        importance_dict[level]=[]
        while True:
            level_node_list=[]
            for item in graph.degree:
                if item[1]<=level:
                    level_node_list.append(item[0])
            graph.remove_nodes_from(level_node_list)
            importance_dict[level].extend(level_node_list)
            if not len(graph.degree):
                return importance_dict
            if min(graph.degree,key=lambda x:x[1])[1]>level:
                break
        level=min(graph.degree,key=lambda x:x[1])[1]
    return importance_dict


def Create_Kshell(G):
    # 计算K-shell分解
    kshells = nx.core_number(G)
    # print(kshells)
    # 计算每个节点的K-shell中心性指标
    k_shell = np.zeros(len(G.nodes))
    nodes = sorted([node for node in G.nodes()])
    assert nodes == [i for i in range(len(G.nodes()))]
    for node in nodes:
        degree = G.degree(node)
        kshell = kshells[node]
        k_shell[node] = kshell  # / degree
    # print(k_shell)

    # ELN 归一化操作
    norm = k_shell.max()
    k_shell = k_shell / norm

    return k_shell

def Create_BC(G):
    bc = nx.betweenness_centrality(G)
    bcvalues = list(bc.values())
    norm = np.array(bcvalues).max()
    ecnorm = bcvalues / norm

    return ecnorm

def Create_Centrality(G, centralities: list):
    """根据需要产生对应的的节点的中心性特征

    Args:
        G (graph): 图
        centralities (list): 存储需要的节点中心性指标的列表
    """
    dc = list()
    hi = list()
    k_shell = list()
    for item in centralities:
        if item == "DC":
            dc = Create_DC(G)
            # print("dc:")
            # print(dc)
        # elif item == "HI":
        #     hi = Create_HI(G)
            # print("HI:")
            # print(hi)
        elif item == "BC":
            bc = Create_BC(G)

        elif item == "Kshell":
            k_shell = Create_Kshell(G)



    centrality_matrix = [np.array(dc), np.array(bc), np.array(k_shell)]
    # print(centrality_matrix)
    centrality_matrix = np.array(centrality_matrix)
    centrality_matrix = centrality_matrix.T
    # print(centrality_matrix)
    # print(centrality_matrix.shape)
    # save_matrix2txt("dataset/temp_matrix/" + G.name + "_centralities.txt", np.array(centrality_matrix),
    #                 save_type="float")
    np.save('./data/features/' + G.name + '_c3.npy', np.array(centrality_matrix))
    return centrality_matrix


def dirget_edge_index(graph_name, is_train):
    if is_train:
        edge_data = np.loadtxt("dataset/" + graph_name + ".txt", skiprows=0,
                               usecols=[0, 1])
    else:
        matrix = mmread(graph_name + ".mtx")
        # 访问行索引
        rows = matrix.row
        # 访问列索引
        cols = matrix.col
        # 访问数据值
        data = matrix.data
        length = int(len(data) / 2)
        edge_data = []
        for i in range(length):
            edg = np.array([rows[i], cols[i]])
            edge_data.extend(edg)
        edge_data = np.array(edge_data).reshape(length, 2)

    edge_list = [list(line) for line in edge_data]
    edge_np = np.array(edge_list)
    edge_np_T = edge_np.T
    edge_np_T_1 = edge_np_T[0]
    edge_np_T_2 = edge_np_T[1]
    edge_index = np.vstack((edge_np_T_1, edge_np_T_2))
    # print(edge_index)
    np.save("./data/features/" + graph_name + "_ei.npy", edge_index)


def Obtain_Centrality(G):
    dc = list()
    hi = list()
    bc = list()
    k_shell = list()

    dc = Create_DC(G)

    # hi = Create_HI(G)

    bc = Create_BC(G)

    k_shell = Create_Kshell(G)

    centrality_matrix = [np.array(dc), np.array(bc), np.array(k_shell)]
    # print(centrality_matrix)
    centrality_matrix = np.array(centrality_matrix)
    centrality_matrix = centrality_matrix.T

    return centrality_matrix


if __name__ == '__main__':
    #BA_1000 = nx.barabasi_albert_graph(1000, 4)
    # name = [ '100_5', '1000_5', '34_karate', '200_3', '1000_4', '200_5', '500_10', '1133_email', '379_netscience']
    name =['polbooks_105']
    for graph_name in name:
        centralities_list = ["DC", "Kshell", "BC"]
        G = nx.Graph()
        G.name = graph_name
        matrix = mmread(graph_name + ".mtx")
        # 访问行索引
        rows = matrix.row
        # 访问列索引
        cols = matrix.col
        # 访问数据值
        data = matrix.data
        length = int(len(data) / 2)
        edge_data = []
        for i in range(length):
            edg = np.array([rows[i],cols[i]])
            edge_data.extend(edg)
        edge_data = np.array(edge_data).reshape(length, 2)

        edge_list = [tuple(line) for line in edge_data]
        nodes_num = int(graph_name.split("_")[1])
        G.add_nodes_from([i for i in range(nodes_num)])
        G.add_edges_from(edge_list)

        dirget_edge_index(graph_name, False)
        # adj_matrix(G)
        # print( nx.adjacency_matrix(G).todense())
        c = Create_Centrality(G, centralities_list)
        # cc = Obtain_Centrality(G)
        # print(cc)
        print(c)
