import copy
import graphviz as gv
import networkx as nx
from DataLoader import load
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


def generate_normal_tree(D, A: dict, cur_node_index):
    """
    建树
    :param D: 训练集
    :param A: 属性字典
    :param cur_node_index: 当前点的索引
    :return: void
    """
    X, y = split_data_and_target(D)
    # 样本都属于同一类别
    if len(set(y)) == 1:
        g.nodes[cur_node_index]['label'] = 'class: {}'.format(y[0])
        return

    # 属性集为空或所有样本的取值都相同
    if len(A) == 0 or count_value_num(X, X[0]) == len(X):
        # 找到出现次数最多的类别
        g.nodes[cur_node_index]['label'] = 'class: {}'.format(collections.Counter(y).most_common(1)[0][0])
        return
    # 选出最优属性
    attribute_index = get_greatest_split_attribute(D, A, solver=solver)
    # 标记当前结点行为
    g.nodes[cur_node_index]['label'] = 'attribute: {}'.format(attribute_index)
    # 对最优属性的每个取值产生一个分支
    for v in A[attribute_index]:
        # 所有数据中在最优属性上取值为v的子集
        Dv = get_Dv(D, attribute_index, v)
        node_num = g.number_of_nodes()
        if len(Dv) == 0:
            # 创建分支结点
            # 将分支结点标记为叶结点，其类别为D中样本最多的类别
            g.add_node(node_num + 1, label='class: {}'.format(collections.Counter(y).most_common(1)[0][0]))
            g.add_edge(cur_node_index, node_num + 1, label='{}'.format(v))
        else:
            # 创建分支结点
            # 分支结点的属性选择需要下一步递归确定
            g.add_node(node_num + 1, label=None, data=Dv)
            g.add_edge(cur_node_index, node_num + 1, label='{}'.format(v))
            new_A = copy.deepcopy(A)
            new_A.pop(attribute_index)
            generate_normal_tree(Dv, new_A, node_num + 1)


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
    :param cur: 当前遍历的结点
    :return: void
    """
    node_label = graph.nodes[cur]['label']
    image.node(str(cur), label=node_label)
    if node_label.startswith('class: '):
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


def get_depth(cur):
    """
    获取结点深度
    :param cur: 当前结点索引 
    :return: void
    """
    # depth[i][0] 为深度
    # depth[i][1] 为结点索引
    for nxt in g.neighbors(cur):
        depth[nxt][0] = depth[cur][0] + 1
        depth[nxt][1] = nxt
        get_depth(nxt)


def generate_post_pruning_tree(g):
    """
    后剪枝
    :param g: 需要剪枝的决策树 
    :return: void 
    """
    # 按照结点深度降序遍历
    for node in depth:
        # 遇到叶结点跳过
        if g.nodes[node[1]]['label'].startswith('class: '):
            continue

        # 考量去掉其子结点后准确率是否提高
        new_g = copy.deepcopy(g)
        for nei in list(new_g.neighbors(node[1])):
            new_g.remove_node(nei)
        Dv = new_g.nodes[node[1]]['data']
        X, y = split_data_and_target(Dv)
        new_g.nodes[node[1]]['label'] = 'class: {}'.format(collections.Counter(y).most_common(1)[0][0])
        # 剪枝后准确率更高
        if get_accuracy(new_g, D_test) >= get_accuracy(g, D_test):
            g = new_g
    return g


data_name = 'lymphography'
solver = 'Gain'

D, class_dicts = load(data_name)
sample_num = len(D)
D_train = D[:int(sample_num * 0.8)]
D_test = D[int(sample_num * 0.8):]

# 创建空图
g = nx.DiGraph()
image = gv.Digraph()
g.add_node(1, label=None, data=D_train)
generate_normal_tree(D_train, class_dicts, 1)
depth = np.ones((g.number_of_nodes() + 1, 2))
get_depth(1)
depth = np.array(sorted(depth, key=lambda d: d[0], reverse=True), dtype=int)
g = generate_post_pruning_tree(g)
generate_image(g, 1)
image.render('../image/post_pruning-{}-{}.gv'.format(solver, data_name))
print(get_accuracy(g, D_test))
