import random
import copy
import time
from scipy.io import mmread

def readGraph_direct(path):
    parentss = { }
    children = { }
    edges = { }
    nodes = set()
    f = open(path, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        row = line.split()
        src = int(row[0])
        dst = int(row[1])
        nodes.add(src)
        nodes.add(dst)
        if children.get(src) is None:
            children[src] = set()
        if parentss.get(dst) is None:
            parentss[dst] = set()
        edges[(src, dst)] = 0
        children[src].add(dst)
        parentss[dst].add(src)
    for edge in edges:
        dst = edge[1]
        #edges[edge] = 1 / len(parentss[dst])
        edges[edge] = 0.1
    return Graph(nodes, edges, children, parentss)

def readGraph_undirect(path):
    parentss = {}
    children = {}
    edges = {}
    nodes = set()
    matrix = mmread(path)
    # 访问行索引
    rows = matrix.row
    # 访问列索引
    cols = matrix.col
    # 访问数据值
    data = matrix.data
    length = int(len(data)/2)
    for i in range(length):
        src = int(rows[i])
        dst = int(cols[i])
        nodes.add(src)
        nodes.add(dst)
        if children.get(src) is None:
            children[src] = set()
        if children.get(dst) is None:
            children[dst] = set()
        if parentss.get(src) is None:
            parentss[src] = set()
        if parentss.get(dst) is None:
            parentss[dst] = set()
        edges[(src, dst)] = 0
        edges[(dst, src)] = 0
        children[src].add(dst)
        children[dst].add(src)
        parentss[src].add(dst)
        parentss[dst].add(src)

    common_value_dict = {}
    for edge in edges:
        dst = edge[1]
        # edges[edge] = 1 / len(parentss[dst])  #注意权重可能等于1
        edges[edge] = 0.1
        # pp = [0.01, 0.05, 0.1, 0.15]
        # reverse_key = tuple(reversed(edge))
        # if reverse_key in common_value_dict:
        #     edges[edge] = common_value_dict[reverse_key]
        # else:
        #     value = random.choice(pp)
        #     edges[edge] = value
        #     common_value_dict[edge] = value
    return Graph(nodes, edges, children, parentss)


class Graph:
    nodes = None
    edges = None
    children = None
    parentss = None
    def __init__(self, nodes, edges, children, parentss):
        self.nodes = nodes
        self.edges = edges
        self.children = children
        self.parentss = parentss
    def get_children(self, node):
        itsChildren = self.children.get(node)
        if itsChildren is None:
            return set()
        return self.children[node]
    def get_parentss(self, node):
        itsParentss = self.parentss.get(node)
        if itsParentss is None:
            return set()
        return self.parentss[node]

def isHappened(prob):
    if prob == 1:
        return True
    if prob == 0:
        return False
    rand = random.random()
    if rand <= prob:
        return True
    else:
        return False

def chunkIt(list, n):
    avg = len(list) / float(n)
    out = []
    last = 0.0
    while last < len(list):
        out.append(list[int(last):int(last + avg)])
        last += avg
    return out

def getSubgraph(graph, inactiveUser):
    nodes = copy.deepcopy(inactiveUser)   #未被激活复制给nodes
    edges = {}
    children = {}
    parentss = {}
    for edge in graph.edges:
        src = edge[0]
        dst = edge[1]
        if src in nodes and dst in nodes:
            edges[edge] = graph.edges[edge]
            if children.get(src) is None:
                children[src] = set()
            if parentss.get(dst) is None:
                parentss[dst] = set()
            children[src].add(dst)
            parentss[dst].add(src)
    return Graph(nodes, edges, children, parentss)

def generate_Node_acceptance(graph):
    nodes_acceptance = {}
    for node in graph.nodes:
        nodes_acceptance[node] = random.random()
    return nodes_acceptance


if __name__ == '__main__':
    path = "lesmis_76.mtx"
    k = 10
    graph = readGraph_undirect(path)
    #graph = readGraph_undirect(path)
    print(graph.edges)
