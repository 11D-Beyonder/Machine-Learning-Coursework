import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm


def Bagging(T):
    clf = []
    for t in range(T):
        # 等概率随机采样
        clf.append(data[np.random.choice(m, size=m)])
    return np.array(clf)


def dis(x1, x2):
    return (x1[0] - x2[0]) * (x1[0] - x2[0]) + (x1[1] - x2[1]) * (x1[1] - x2[1])


def predict(clf, x):
    ans = 0
    for t in range(len(clf)):
        nearest = None
        min_dis = np.inf
        # 对clf[t]找1NN
        for watermelon in clf[t]:
            temp = dis(watermelon, x)
            if temp < min_dis:
                nearest = watermelon
                min_dis = temp
        ans += nearest[-1]
    return 1 if ans > 0 else -1


if __name__ == '__main__':
    data = pd.read_csv('watermelon3.0a.csv', header=0).values
    X, y = data[:, :-1], data[:, -1]
    m, n = X.shape

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
            if i % 1000 == 0:
                print(i)
        ZZ = ZZ.reshape(XX.shape)
        # 划分边界
        # https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        plt.contourf(XX, YY, ZZ, alpha=0.4, cmap=cm.gist_ncar)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], label='好瓜', color='red', edgecolors='k')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], label='坏瓜', color='green', edgecolors='k')

        plt.title('{}个基学习器'.format(T))
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        print(T)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.tight_layout()
    plt.show()
