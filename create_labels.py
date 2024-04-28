import networkx as nx
import numpy as np
from tqdm import tqdm
import random
import copy
import graphs
from scipy.io import mmread
# import centrality as cc
# TO 主要实现的就是训练网络节点的标签生成。

def localSearch(graph, root):
    localSet = set()
    queue = []
    queue.append(root)
    localSet.add(root)
    while len(queue) != 0:
        current_node = queue.pop(0)
        children = graph.get_children(current_node)
        for child in children:
            if child not in localSet:
                rate = graph.edges[(child, current_node)]
                if graphs.isHappened(rate):
                    localSet.add(child)
                    queue.append(child)
    return localSet

def dirlocalSearch(graph, root):
    localSet = set()
    queue = []
    queue.append(root)
    localSet.add(root)
    while len(queue) != 0:
        current_node = queue.pop(0)
        children = graph.get_children(current_node)
        for child in children:
            if child not in localSet:
                rate = graph.edges[(current_node, child)]
                if graphs.isHappened(rate):
                    localSet.add(child)
                    queue.append(child)
    return localSet

def IC(graph, graph_name):
    print("---- start create labels ----")
    simulations = 200
    influence = list()
    for i in graph.nodes:
        count = 0
        for sim in range(simulations):
            inf = len(localSearch(graph, i))
            count += inf
        aveinf = count / simulations
        influence.append(aveinf)
    # print(graph.nodes)
    print(influence)
    np.save("./data/labels/" + graph_name + ".npy", np.array(influence))
    print("---- end create labels ----")

#SIR
def update_node_status(graph, node, beta, gamma):
    """
    更新节点状态
    :param graph: 网络图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
    if graph.nodes[node]['status'] == 'I':
        p = random.random()
        if p < gamma:
            graph.nodes[node]['status'] = 'R'
    # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
    if graph.nodes[node]['status'] == 'S':
        # 获取当前节点的邻居节点
        # 无向图：G.neighbors(node)
        # 有向图：G.predecessors(node)，前驱邻居节点，即指向该节点的节点；G.successors(node)，后继邻居节点，即该节点指向的节点。
        neighbors = list(graph.neighbors(node))
        # 对当前节点的邻居节点进行遍历
        for neighbor in neighbors:
            # 邻居节点中存在 感染者(I)，则该节点有概率被感染为 感染者(I)
            if graph.nodes[neighbor]['status'] == 'I':
                p = random.random()
                if p < beta:
                    graph.nodes[node]['status'] = 'I'
                    break


def count_node(graph):
    """
    计算当前图内各个状态节点的数目
    :param graph: 输入图
    :return: 各个状态（S、I、R）的节点数目
    """
    s_num, i_num, r_num = 0, 0, 0
    for node in graph:
        if graph.nodes[node]['status'] == 'S':
            s_num += 1
        elif graph.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num


def SIR_network(graph, source, beta, gamma, step):
    """
    获得感染源的节点序列的SIR感染情况
    :param graph: networkx创建的网络
    :param source: 需要被设置为感染源的节点Id所构成的序列
    :param beta: 感染率
    :param gamma: 免疫率
    :param step: 迭代次数
    """
    n = graph.number_of_nodes()  # 网络节点个数
    sir_values = []  # 存储每一次迭代后网络中感染节点数I+免疫节点数R的总和
    # 初始化节点状态
    for node in graph:
        graph.nodes[node]['status'] = 'S'  # 将所有节点的状态设置为 易感者（S）
    # 设置初始感染源
    for node in source:
        graph.nodes[node]['status'] = 'I'  # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values.append(len(source) / n)
    # 开始迭代感染
    for s in range(step):
        # 针对对每个节点进行状态更新以完成本次迭代
        for node in graph:
            update_node_status(graph, node, beta, gamma)  # 针对node号节点进行SIR过程
        s, i, r = count_node(graph)  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
        sir = (i + r) / n  # 该节点的sir值为迭代结束后 感染节点数i+免疫节点数r
        sir_values.append(sir)  # 将本次迭代的sir值加入数组
    return sir_values

def SIR(graph, graph_name):
    print("---- start create labels ----")
    simulations = 300
    degree = dict(nx.degree(graph))
    # 平均度为所有节点度之和除以总节点数
    ave_degree =  sum(degree.values()) / len(graph)
    # 计算节点的二阶平均度
    second_order_avg_degree = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        second_order_degrees = [graph.degree(neighbor) for neighbor in neighbors]
        if len(second_order_degrees) > 0:
            second_order_avg_degree.append(sum(second_order_degrees) / len(second_order_degrees))  # 计算邻居节点的平均度
    # 计算二阶平均度
    second_order_avg_degree = sum(second_order_avg_degree) / len(second_order_avg_degree)
    beta = ave_degree / second_order_avg_degree # 感染率
    gamma = 0.1  # 免疫率
    step = 20  # 迭代次数
    influence = list()
    for i in graph.nodes:
        count = 0
        for sim in range(simulations):
            inf = sum(SIR_network(graph,[i], beta, gamma, step))
            count += inf
        aveinf = count / simulations
        influence.append(aveinf)
    # print(graph.nodes)
    print(influence)
    np.save("./data/labels/" + graph_name + ".npy", np.array(influence))
    print("---- end create labels ----")

if __name__ == '__main__':

    # name = ['100_5', '34_karate', '1133_email', '379_netscience']
    # for graph_name in name:
    #     G = cc.create_graph(graph_name)
    #     simulate(G, 100, 0.1, graph_name)

    graph_name = "lesmis_77"
    # graph_name = "polbooks_105"
    # path = str(graph_name) + '.mtx'
    # graph = graphs.readGraph_undirect(path)
    # IC(graph, graph_name)

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
        edg = np.array([rows[i], cols[i]])
        edge_data.extend(edg)
    edge_data = np.array(edge_data).reshape(length, 2)

    edge_list = [tuple(line) for line in edge_data]
    nodes_num = int(graph_name.split("_")[1])
    G.add_nodes_from([i for i in range(nodes_num)])
    G.add_edges_from(edge_list)
    SIR(G, graph_name)