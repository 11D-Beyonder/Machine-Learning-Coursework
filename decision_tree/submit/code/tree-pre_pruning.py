import copy
from queue import Queue
import graphviz as gv
import networkx as nx
from DataLoader import load
from util import *


def predict(graph, x):
    """
    预测样本x的值
    :param graph: 构建好的决策树
    :param x: 样本
    :return: 样本预测值
    """
    cur = 1
    while True:
        node_label = graph.nodes[cur]['label']
        if node_label.startswith('class: '):
            return node_label
        else:
            attribute_index = int(node_label.replace('attribute: ', ''))
            for nei in graph.neighbors(cur):
                if graph.get_edge_data(cur, nei)['label'] == str(x[attribute_index]):
                    cur = nei
                    break


def generate_image(graph, cur):
    """
    用graphviz画树
    :param graph: 构建好的决策树
    :param cur: 当前遍历的节点
    :return: void
    """
    node_label = graph.nodes[cur]['label']
    image.node(str(cur), label=node_label)
    if node_label.startswith('class: '):
        # 到叶节点
        return
    else:
        for nei in graph.neighbors(cur):
            image.edge(str(cur), str(nei), label=graph.get_edge_data(cur, nei)['label'])
            generate_image(graph, nei)


def get_accuracy(g, D):
    hits = 0
    X, y = split_data_and_target(D)
    for i in range(len(y)):
        if predict(g, X[i]).replace('class: ', '') == str(y[i]):
            hits += 1
    return hits / len(y)


def get_greatest_split_attribute(D, A: dict, solver='Gini_index'):
    """
    找到最优的分类器
    :param D: 训练集
    :param A: 属性集
    :param solver: 评估函数
    :return: 最优的属性索引
    """
    if solver == 'Gain':
        attribute_index = -1
        ans = -1
        for a in A.keys():
            gain = Gain(D, a, class_dicts)
            if gain > ans:
                ans = gain
                attribute_index = a
        return attribute_index
    elif solver == 'Gini_index':
        attribute_index = -1
        ans = 1000000000000000000
        for a in A.keys():
            gini_index = Gini_index(D, a, class_dicts)
            if gini_index < ans:
                ans = gini_index
                attribute_index = a
        return attribute_index
    else:
        raise Exception('solver不正确')


def generate_pre_pruning_tree(ini_D, ini_A):
    """
    预剪枝建树
    :param D: 训练集
    :param A: 属性字典
    :param cur_node_index: 当前点的索引
    :return: void
    """
    ini_X, ini_y = split_data_and_target(ini_D)
    g.add_node(1, label='class: {}'.format(collections.Counter(ini_y).most_common(1)[0][0]))
    q = Queue()
    q.put([1, ini_D, ini_A])
    while not q.empty():
        cur = q.get_nowait()
        cur_node_index, D, A = cur[0], cur[1], cur[2]
        X, y = split_data_and_target(D)
        attribute_index = get_greatest_split_attribute(D, A, solver=solver)

        stop_g = copy.deepcopy(g)
        stop_g.nodes[cur_node_index]['label'] = 'class: {}'.format(collections.Counter(y).most_common(1)[0][0])

        divide_g = copy.deepcopy(g)
        divide_g.nodes[cur_node_index]['label'] = 'attribute: {}'.format(attribute_index)
        # 遍历每一个可能取值v生成子节点

        for v in class_dicts[attribute_index]:
            Dv = get_Dv(D, attribute_index, v)
            node_num = divide_g.number_of_nodes()
            if len(Dv) == 0:
                divide_g.add_node(node_num + 1, label='class: {}'.format(collections.Counter(y).most_common(1)[0][0]))
                divide_g.add_edge(cur_node_index, node_num + 1, label='{}'.format(v))
            else:
                Xv, yv = split_data_and_target(Dv)
                divide_g.add_node(node_num + 1, label='class: {}'.format(collections.Counter(yv).most_common(1)[0][0]))
                divide_g.add_edge(cur_node_index, node_num + 1, label='{}'.format(v))

        if get_accuracy(stop_g, D_test) >= get_accuracy(divide_g, D_test):
            continue

        g.nodes[cur_node_index]['label'] = 'attribute: {}'.format(attribute_index)
        for v in class_dicts[attribute_index]:
            Dv = get_Dv(D, attribute_index, v)
            node_num = g.number_of_nodes()
            if len(Dv) == 0:
                g.add_node(node_num + 1, label='class: {}'.format(collections.Counter(y).most_common(1)[0][0]))
                g.add_edge(cur_node_index, node_num + 1, label='{}'.format(v))
            else:
                Xv, yv = split_data_and_target(Dv)
                g.add_node(node_num + 1, label='class: {}'.format(collections.Counter(yv).most_common(1)[0][0]))
                g.add_edge(cur_node_index, node_num + 1, label='{}'.format(v))
                new_A = copy.deepcopy(A)
                new_A.pop(attribute_index)
                q.put([node_num + 1, Dv, new_A])


data_name = 'balance-scale'
solver = 'Gini_index'

D, class_dicts = load(data_name)
sample_num = len(D)
D_train = D[:int(sample_num * 0.8)]
D_test = D[int(sample_num * 0.8):]

# 创建空图
g = nx.DiGraph()
image = gv.Digraph()

generate_pre_pruning_tree(D_train, class_dicts)
generate_image(g, 1)
image.render('../image/pre_pruning-{}-{}.gv'.format(solver, data_name))
print(get_accuracy(g, D_test))
