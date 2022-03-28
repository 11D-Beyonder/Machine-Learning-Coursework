import copy
from queue import Queue
import graphviz as gv
import networkx as nx
import numpy as np

from util import *


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


def generate_normal_tree(ini_D, ini_A, MaxNode):
    """
    建树
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

        # 样本都属于同一类别
        if len(set(y)) == 1:
            g.nodes[cur_node_index]['label'] = 'class: {}'.format(y[0])
            continue

        # 属性集为空或所有样本的取值都相同
        if len(A) == 0 or count_value_num(X, X[0]) == len(X):
            # 找到出现次数最多的类别
            g.nodes[cur_node_index]['label'] = 'class: {}'.format(collections.Counter(y).most_common(1)[0][0])
            continue

        attribute_index = get_greatest_split_attribute(D, A, solver='Gain')

        if g.number_of_nodes() + len(A[attribute_index]) > MaxNode:
            return

        g.nodes[cur_node_index]['label'] = 'attribute: {}'.format(attribute_index)

        for v in A[attribute_index]:
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
    if node_label.startswith('class'):
        image.node(str(cur), label=node_label, fontname="Microsoft YaHei")
        return
    else:
        node_label = node_label.replace('attribute: ', '')
        node_label = attribute_name[int(node_label)]
        image.node(str(cur), label=node_label, fontname="Microsoft YaHei")
        for nei in graph.neighbors(cur):
            edge_label = graph.get_edge_data(cur, nei)['label']
            image.edge(str(cur), str(nei), label=edge_label, fontname="Microsoft YaHei")
            generate_image(graph, nei)


def get_accuracy(g, D):
    hits = 0
    X, y = split_data_and_target(D)
    for i in range(len(y)):
        if predict(g, X[i]).replace('class: ', '') == str(y[i]):
            hits += 1
    return hits / len(y)


D_train = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['青绿', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']

])
D_test = np.array([
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜']
])
class_dicts = {
    0: {'青绿', '乌黑', '浅白'},
    1: {'蜷缩', '稍蜷', '硬挺'},
    2: {'浊响', '沉闷', '清脆'},
    3: {'清晰', '模糊', '稍糊'},
    4: {'凹陷', '稍凹', '平坦'},
    5: {'硬滑', '软粘'}
}
attribute_name = np.array(['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])

# 创建空图
g = nx.DiGraph()
image = gv.Digraph()

generate_normal_tree(D_train, class_dicts, 5)
generate_image(g, 1)
image.render('../image/limit_MaxNode-Gain-watermelon2.0.gv')
print(get_accuracy(g, D_test))
