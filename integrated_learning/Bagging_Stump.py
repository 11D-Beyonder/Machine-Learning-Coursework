"""
https://blog.csdn.net/aliceyangxi1987/article/details/74625962
https://www.cnblogs.com/NoNameIsBeginning/p/13733146.html
以决策树桩为基学习器
训练Bagging集成
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm


def split_X_y(D):
    return (D[:, :-1], D[:, -1]) if len(D) > 0 else (np.array([]), np.array([]))


def cal_ent(D):
    if len(D) == 0:
        return 0
    X, y = split_X_y(D)
    p_good = len(y[y == 1]) / len(D)
    p_bad = len(y[y == -1]) / len(D)
    m1 = np.log2(p_good) if p_good != 0 else 0
    m2 = np.log2(p_bad) if p_bad != 0 else 0
    return -p_good * m1 - p_bad * m2


def split_sample(D, feature_index, point):
    """
    根据属性的分割点将集合分割为子集
    :param D:
    :param feature_index: 属性索引
    :param point:
    :return: 根据特定属性值分割后的集合
    """
    D_ge = []
    D_le = []
    for i in range(m):
        if D[i][feature_index] <= point:
            D_le.append(D[i])
        else:
            D_ge.append(D[i])
    D_le, D_ge = np.array(D_le), np.array(D_ge)
    return D_le, D_ge


def build_stump(D):
    """
    构造决策树桩，找到最合适的分割属性，及其分割点。
    :param D: 样本集合
    :return: 最优的分割属性和分割点
    """
    max_gain = -np.inf
    best_split_point = None
    best_feature = None
    for feature_index in range(n):
        X, y = split_X_y(D)
        feature_values = X[:, feature_index]
        sorted_feature_values = np.sort(feature_values)
        # 得到分割点
        split_points = [(sorted_feature_values[i] + sorted_feature_values[i + 1]) / 2 for i in range(m - 1)]
        for point in split_points:
            # 得到分割点两侧的子集
            D_le, D_ge = split_sample(D, feature_index, point)
            # 计算信息增益Gain
            gain = cal_ent(D) - len(D_le) / len(D) * cal_ent(D_le) - len(D_ge) / len(D) * cal_ent(D_ge)
            if gain > max_gain:
                max_gain = gain
                best_split_point = point
                best_feature = feature_index

    D1, D2 = split_sample(D, best_feature, best_split_point)
    _, y1 = split_X_y(D1)
    _, y2 = split_X_y(D2)

    return best_feature, best_split_point, \
           1 if len(y1[y1 == 1]) > len(y1[y1 == -1]) else -1, \
           1 if len(y2[y2 == 1]) > len(y2[y2 == -1]) else -1


def Bagging(T):
    clf = np.empty(T, dtype=dict)
    for t in range(T):
        # 以等概率采样
        D = data[np.random.choice(m, size=m)]

        clf[t] = {}
        clf[t]['feature_index'], clf[t]['point'], clf[t]['type1'], clf[t]['type2'] = build_stump(D)
    return clf


def predict(clf, x):
    res = 0
    for t in range(len(clf)):
        res += clf[t]['type1'] if x[clf[t]['feature_index']] <= clf[t]['point'] else clf[t]['type2']
    return 1 if res > 0 else -1


if __name__ == '__main__':
    data = pd.read_csv('watermelon3.0a.csv', header=0).values
    m, n = data.shape
    n -= 1

    plt.figure(figsize=(17, 4))
    for p, T in zip([1, 2, 3], [3, 5, 11]):
        model = Bagging(T)

        plt.subplot(1, 3, p)
        # 预测网格上的结果
        step = 1000
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
        X, y = split_X_y(data)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], label='好瓜', color='red', edgecolors='k')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], label='坏瓜', color='green', edgecolors='k')

        for i in range(len(model)):
            # 得到分界点
            split_point = [model[i]['point']] * step
            # 画横竖线
            if model[i]['feature_index'] == 0:
                plt.plot(split_point, yy)
            else:
                plt.plot(xx, split_point)
        plt.title('{}个基学习器'.format(T))
        plt.xlabel('密度')
        plt.ylabel('含糖率')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.tight_layout()
    plt.show()
