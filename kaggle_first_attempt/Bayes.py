import numpy as np

import data_loader


def get_P(c, x_i, i):
    """
    假定 p(x_i|c)~N(\mu,\sigma^2)
    :param c: 种类
    :return: 后验概率
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma[c][i]) * np.exp(-((x_i - mu[c][i]) ** 2 / (2 * sigma[c][i] ** 2)))


def predict(sample):
    max_value = -1
    ans = None
    for c in target_names:
        nb = P[c]
        for i in range(d):
            nb *= get_P(c, sample[i], i)
        if nb > max_value:
            max_value = nb
            ans = c

    return ans


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_loader.data_fruit(split_train_test=True)
    # 数据个数
    N = len(X_train)
    # 维度
    d = len(X_train[0])
    mu = {}
    sigma = {}
    target_names = ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY']
    P = {}
    for c in target_names:
        mu.update({c: np.zeros(d)})
        sigma.update({c: np.zeros(d)})
        # 计算种类c的频率
        P.update({c: (len(y_train[y_train == c]) + 1) / (len(y_train) + 7)})
        # mu[c][i]和sigma[c][i]分别是第c类样本在第i个属性上取值的均值和方差
        for i in range(d):
            mu[c][i] = np.mean(X_train[y_train == c, i])
            sigma[c][i] = np.std(X_train[y_train == c, i], ddof=1)

    accuracy = 0
    for test, label in zip(X_test, y_test):
        if label == predict(test):
            accuracy += 1
    print(accuracy / len(y_test))
