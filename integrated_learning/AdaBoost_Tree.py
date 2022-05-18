import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm


def split_X_y(data):
    """
    分割为属性集合和真实标签
    """
    return (data[:, :-1], data[:, -1]) if len(data) > 0 else (np.array([]), np.array([]))


def split_sample(data, feature_index, point, D):
    """
    根据属性的分割点将集合分割为子集
    :param D: 各个样本的权重
    :param data: 样本集合
    :param feature_index: 属性索引
    :param point: 分割点
    :return: 根据特定属性值分割后的集合
    """
    data_ge = []
    data_le = []
    D_ge = []
    D_le = []
    for i in range(len(data)):
        if data[i][feature_index] <= point:
            data_le.append(data[i])
            D_le.append(D[i])
        else:
            data_ge.append(data[i])
            D_ge.append(D[i])
    return np.array(data_le), np.array(data_ge), np.array(D_le), np.array(D_ge)


def cal_ent(data, D):
    if len(data) == 0:
        return 0
    X, y = split_X_y(data)
    # 好瓜比例
    p_good = np.sum(D[y == 1])
    # 坏瓜比例
    p_bad = np.sum(D[y == -1])
    m1 = np.log2(p_good) if p_good != 0 else 0
    m2 = np.log2(p_bad) if p_bad != 0 else 0
    return -p_good * m1 - p_bad * m2


def build_tree(cur: dict, data, D, deep=2):
    """
    树结点
    label: None 或 种类标签
    point: 分割点
    feature_index: 分割属性
    l_child: 左儿子
    r_child: 右儿子

    构建基于Gain的决策树
    :param deep: 决策树最大深度
    :param cur: 当前结点
    :param data: 当前结点上的样本集合
    :param D: 分布（各个样本的权重）
    :return: 根节点cur
    """
    X, y = split_X_y(data)
    if deep == 1 or len(set([tuple(watermelon) for watermelon in X])) == 1:
        # 所有样本各个属性值皆相等
        cur['label'] = 1 if len(y[y == 1]) > len(y[y == -1]) else -1
        cur['l_child'] = cur['r_child'] = cur['feature_index'] = cur['point'] = None
        return cur

    max_gain = -np.inf
    best_split_point = None
    best_feature = None

    for feature_index in range(n):
        X, y = split_X_y(data)
        feature_values = X[:, feature_index]
        sorted_feature_values = np.sort(feature_values)
        # 得到分割点
        split_points = [(sorted_feature_values[i] + sorted_feature_values[i + 1]) / 2 for i in range(len(X) - 1)]
        for point in split_points:
            # 得到分割点两侧的子集
            data_le, data_ge, D_le, D_ge = split_sample(data, feature_index, point, D)
            # 计算信息增益Gain
            gain = cal_ent(data, D) - np.sum(D_le) * cal_ent(data_le, D_le) - np.sum(D_ge) * cal_ent(data_ge, D_ge)

            if gain > max_gain:
                max_gain = gain
                best_split_point = point
                best_feature = feature_index

    # 根据分割点生成分支
    data1, data2, D1, D2 = split_sample(data, best_feature, best_split_point, D)
    cur['label'], cur['point'], cur['feature_index'] = None, best_split_point, best_feature
    cur['l_child'], cur['r_child'] = {}, {}

    if len(data1) == 0:
        cur['l_child']['label'] = 1 if len(y[y == 1]) > len(y[y == -1]) else -1
    else:
        cur['l_child'] = build_tree(cur['l_child'], data1, D1, deep - 1)
    if len(data2) == 0:
        cur['r_child']['label'] = 1 if len(y[y == 1]) > len(y[y == -1]) else -1
    else:
        cur['r_child'] = build_tree(cur['r_child'], data2, D2, deep - 1)

    return cur


def tree_predict(node, x):
    if node['label'] is None:
        if x[node['feature_index']] <= node['point']:
            return tree_predict(node['l_child'], x)
        else:
            return tree_predict(node['r_child'], x)
    else:
        return node['label']


def AdaBoost(T):
    D = np.ones(m) / m
    clf = np.empty(T, dtype=dict)
    for t in range(T):
        clf[t] = {}

        clf[t]['tree'] = build_tree(clf[t], dataset, D)
        # 统计错误率
        error = 0
        h_t = np.empty(m)
        for i in range(m):
            h_t[i] = tree_predict(clf[t]['tree'], dataset[i, :-1])
            if h_t[i] != dataset[i, -1]:
                error += D[i]

        if error > 0.5:
            break
        alpha_t = 0.5 * np.log(1 / error - 1)
        clf[t]['alpha'] = alpha_t
        temp = np.exp(-alpha_t * h_t * dataset[:, -1])
        D = D * temp / np.dot(D, temp)

    return clf


def predict(clf, x):
    res = 0
    for t in range(len(clf)):
        res += clf[t]['alpha'] * tree_predict(clf[t]['tree'], x)
    return 1 if res >= 0 else -1


if __name__ == '__main__':
    dataset = pd.read_csv('watermelon3.0a.csv', header=0).values
    m, n = dataset.shape
    n -= 1

    plt.figure(figsize=(17, 4))
    for p, T in zip([1, 2, 3], [3, 5, 11]):
        model = AdaBoost(T)

        plt.subplot(1, 3, p)
        # 预测网格上的结果
        step = 100
        xx = np.linspace(0.2, 0.8, step)
        yy = np.linspace(0, 0.5, step)
        XX, YY = np.meshgrid(xx, yy)
        points = np.c_[XX.ravel(), YY.ravel()]
        ZZ = np.empty(len(points))
        for i in range(len(points)):
            ZZ[i] = predict(model, points[i])
        ZZ = ZZ.reshape(XX.shape)
        # 划分边界
        # https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        plt.contourf(XX, YY, ZZ, alpha=0.4, cmap=cm.gist_ncar)
        plt.scatter(dataset[dataset[:, -1] == 1, 0], dataset[dataset[:, -1] == 1, 1], label='好瓜', color='red',
                    edgecolors='k')
        plt.scatter(dataset[dataset[:, -1] == -1, 0], dataset[dataset[:, -1] == -1, 1], label='坏瓜', color='green',
                    edgecolors='k')

        plt.title('{}个基学习器'.format(T))
        plt.xlabel('密度')
        plt.ylabel('含糖率')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.tight_layout()
    plt.show()
