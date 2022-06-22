import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import data_loader


def kNN(k, X, y, sample, solver='euclidean'):
    dis = []
    n = len(y)
    for i in range(n):
        # 存距离和真实标记
        val = None
        if solver == 'euclidean':
            val = np.sqrt(np.dot(X[i] - sample, X[i] - sample))
        elif solver == 'manhattan':
            val = np.sum(np.abs(X[i] - sample))
        elif solver == 'mahalanobis':
            val = np.sqrt((sample - X[i]).T @ Cov @ (sample - X[i]))
        dis.append((val, y[i]))
    # 按照距离排序
    dis.sort()
    # 取前k个
    return dis[:k]


if __name__ == '__main__':
    fig = plt.figure()

    X_train, y_train, X_test, y_test = data_loader.data_fruit(split_train_test=True)
    Cov = np.linalg.inv(np.cov(X_train, rowvar=False))
    sns.set_style('darkgrid')

    data = {'k': [], 'accuracy': [], 'solver': []}

    for k in range(1, 11):
        print(k)
        # 找到前k近的邻居
        accuracy = 0
        for test, label in zip(X_test, y_test):
            neighbors = kNN(k, X_train, y_train, test, solver='mahalanobis')
            # 统计出现次数最多的
            cnt = {}
            for d, clazz in neighbors:
                if cnt.get(clazz) is None:
                    cnt.update({clazz: 1})
                else:
                    cnt[clazz] += 1
            ans = None
            max_count = -1
            for clazz, count in cnt.items():
                if count > max_count:
                    ans = clazz

            if label == ans:
                accuracy += 1

        accuracy /= len(y_test)
        print(accuracy)
        data['k'].append(k)
        data['accuracy'].append(accuracy)
        data['solver'].append('mahalanobis')

    for k in range(1, 11):
        print(k)
        # 找到前k近的邻居
        accuracy = 0
        for test, label in zip(X_test, y_test):
            neighbors = kNN(k, X_train, y_train, test, solver='manhattan')
            # 统计出现次数最多的
            cnt = {}
            for d, clazz in neighbors:
                if cnt.get(clazz) is None:
                    cnt.update({clazz: 1})
                else:
                    cnt[clazz] += 1
            ans = None
            max_count = -1
            for clazz, count in cnt.items():
                if count > max_count:
                    ans = clazz

            if label == ans:
                accuracy += 1

        accuracy /= len(y_test)
        print(accuracy)
        data['k'].append(k)
        data['accuracy'].append(accuracy)
        data['solver'].append('manhattan')

    for k in range(1, 11):
        print(k)
        # 找到前k近的邻居
        accuracy = 0
        for test, label in zip(X_test, y_test):
            neighbors = kNN(k, X_train, y_train, test, solver='euclidean')
            # 统计出现次数最多的
            cnt = {}
            for d, clazz in neighbors:
                if cnt.get(clazz) is None:
                    cnt.update({clazz: 1})
                else:
                    cnt[clazz] += 1
            ans = None
            max_count = -1
            for clazz, count in cnt.items():
                if count > max_count:
                    ans = clazz

            if label == ans:
                accuracy += 1

        accuracy /= len(y_test)
        print(accuracy)
        data['k'].append(k)
        data['accuracy'].append(accuracy)
        data['solver'].append('euclidean')

    res = pd.DataFrame(data)
    print(res)
    sns.lineplot(x="k", y="accuracy", hue="solver", style="solver", markers=True, dashes=True, data=res, linestyle='-')
    plt.tight_layout()
    plt.show()
