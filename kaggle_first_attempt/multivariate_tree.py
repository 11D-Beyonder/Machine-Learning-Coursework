import collections

import graphviz as gv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

import data_loader


def split_data_and_target(D):
    """
    :param D: 训练集
    :return: 拆分为参数和真实值
    """
    if len(D) == 0:
        return [], []
    else:
        return D[:, :-1], D[:, -1]


def generate_normal_tree(X, y, cur_node_index):
    """
    建树
    :param X: 样本
    :param y: 真实标记
    :param cur_node_index: 当前点的索引
    :return: void
    """
    # 样本都属于同一类别
    if len(set(y)) == 1:
        g.nodes[cur_node_index]['label'] = 'class: {}'.format(y[0])
        return

    # 进行线性分类
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    g.nodes[cur_node_index]['w'] = w
    g.nodes[cur_node_index]['b'] = b
    g.nodes[cur_node_index]['label'] = 'w={},b={}'.format(w, b)
    # 得到分类后的子集
    X1, y1 = [], []
    X2, y2 = [], []
    for i in range(len(y)):
        if np.dot(w, X[i]) + b > 0:
            X1.append(X[i])
            y1.append(y[i])
        else:
            X2.append(X[i])
            y2.append(y[i])

    X1 = np.array(X1)
    X2 = np.array(X2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    if len(y1) == 0 or len(y2) == 0:
        # 不可分类
        g.nodes[cur_node_index]['label'] = 'class: {}'.format(collections.Counter(y).most_common(1)[0][0])
        return

    # 产生分支节点
    node_num = g.number_of_nodes()
    g.add_node(node_num + 1, label=None, w=None, b=None)
    g.add_node(node_num + 2, label=None, w=None, b=None)
    g.add_edge(cur_node_index, node_num + 1, label='wx+b>0')
    g.add_edge(cur_node_index, node_num + 2, label='wx+b<=0')
    generate_normal_tree(X1, y1, node_num + 1)
    generate_normal_tree(X2, y2, node_num + 2)


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
            w = graph.nodes[cur]['w']
            b = graph.nodes[cur]['b']
            val = np.dot(w, x) + b
            for nei in graph.neighbors(cur):
                if graph.get_edge_data(cur, nei)['label'] == 'wx+b>0' and val > 0:
                    cur = nei
                    break
                elif graph.get_edge_data(cur, nei)['label'] == 'wx+b<=0' and val <= 0:
                    cur = nei
                    break


def generate_image(graph, cur):
    """
    用graphviz画树
    :param graph: 构建好的决策树
    :param cur: 当前遍历的结点
    :return: void
    """
    node_label = graph.nodes[cur]['label']
    image.node(str(cur), label=node_label)
    if node_label.startswith('class'):
        # 到叶结点
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


D_train, D_test = data_loader.data_fruit(split_train_test=True, split_sample_target=False)
# 创建空图
g = nx.DiGraph()
image = gv.Digraph(encoding='UTF-8')

g.add_node(1, label=None, w=None, b=None)
X_train, y_train = split_data_and_target(D_train)
generate_normal_tree(X_train, y_train, 1)
generate_image(g, 1)
image.render('image/multivariate_tree.gv')
print(get_accuracy(g, D_test))
