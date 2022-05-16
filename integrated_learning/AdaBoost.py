import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

"""
这里题目有点问题，如果以不剪枝决策树为基学习器，
可以生成一个完美符合数据的决策树，此时AdaBoost就没意义了，
因为第一颗树错误率就为0了，样本权重也不会发生改变。
"""


def cal_error(split_point, sign, D, feature_values):
    error = 0
    for i in range(m):
        if sign == 'lower':
            # 小于等于split_point为好瓜，大于solit_point为坏瓜，预测正确。
            # 否则就预测错误。
            if feature_values[i] <= split_point and y[i] == -1:
                error += D[i]
            elif feature_values[i] > split_point and y[i] == 1:
                error += D[i]
        else:
            # 大于split_point为好瓜，小于等于solit_point为坏瓜，预测正确。
            # 否则就预测错误。
            if feature_values[i] > split_point and y[i] == -1:
                error += D[i]
            elif feature_values[i] <= split_point and y[i] == 1:
                error += D[i]
    return error


def get_best_split_point(feature_index, D):
    feature_values = X[:, feature_index]
    sorted_feature_values = np.sort(feature_values)
    split_points = [(sorted_feature_values[i] + sorted_feature_values[i + 1]) / 2 for i in range(m - 1)]
    best_split_point = None
    min_error = np.inf
    best_sign = None
    for point in split_points:
        for sign in ['lower', 'greater']:
            error = cal_error(point, sign, D, feature_values)
            if error < min_error:
                min_error = error
                best_split_point = point
                best_sign = sign
    return best_split_point, best_sign, min_error


def build_best_stump(D):
    best_split_point = None
    min_error = np.inf
    best_sign = None
    best_feature = None
    for feature_index in range(n):
        split_point, sign, error = get_best_split_point(feature_index, D)
        if error < min_error:
            min_error = error
            best_feature = feature_index
            best_split_point = split_point
            best_sign = sign
            print(best_feature, best_sign, best_split_point, min_error)
    return best_feature, best_split_point, best_sign, min_error


def cal_ht(stump, x):
    if stump['sign'] == 'lower':
        return 1 if x[stump['feature_index']] <= stump['point'] else -1
    else:
        return 1 if x[stump['feature_index']] > stump['point'] else -1


def AdaBoost(T):
    D = np.ones(m) / m
    clf = np.empty(T, dtype=dict)
    for t in range(T):
        clf[t] = {}
        feature, point, sign, error = build_best_stump(D)
        if error > 0.5:
            break
        alpha_t = 0.5 * np.log(1 / error - 1)
        stump = {'feature_index': feature, 'point': point, 'sign': sign}
        h_t = np.empty(m)
        for i in range(m):
            h_t[i] = cal_ht(stump, X[i])

        clf[t]['alpha'] = alpha_t
        clf[t]['stump'] = stump
        temp = np.exp(-alpha_t * h_t * y)
        D = D * temp / np.dot(D, temp)

    return clf


def predict(clf, x):
    res = 0
    for t in range(len(clf)):
        res += clf[t]['alpha'] * cal_ht(clf[t]['stump'], x)
    return 1 if res >= 0 else -1


if __name__ == '__main__':
    data = pd.read_csv('watermelon3.0a.csv', header=0).values
    X = data[:, :-1]
    m, n = X.shape
    y = data[:, -1]
    plt.figure(figsize=(17, 4))
    for p, T in zip([1, 2, 3], [3, 5, 11]):
        model = AdaBoost(T)

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
        plt.scatter(X[y == 1, 0], X[y == 1, 1], label='好瓜', color='red', edgecolors='k')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], label='坏瓜', color='green', edgecolors='k')

        for i in range(len(model)):
            # 得到分界点
            split_point = [model[i]['stump']['point']] * step
            # 画横竖线
            if model[i]['stump']['feature_index'] == 0:
                plt.plot(split_point, yy)
            else:
                plt.plot(xx, split_point)

        plt.xlabel('密度')
        plt.ylabel('含糖率')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.tight_layout()
    plt.show()
