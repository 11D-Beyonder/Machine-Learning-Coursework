import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, theta, gamma, v, w):
    alpha = np.dot(x, v)
    b = sigmoid(alpha - gamma)
    beta = np.dot(b, w)
    y_hat = sigmoid(beta - theta)
    return y_hat


def E(X, Y, theta, gamma, v, w):
    ans = 0
    for k in range(m):
        x = X[k]
        y = Y[k]
        y_hat = predict(x, theta, gamma, v, w)
        ans += np.linalg.norm(y - y_hat) ** 2
    return ans / m


data = np.array(pd.read_csv("../data/watermelon3.0.csv").iloc[:, 1:])

value_dict = {
    "浅白": 1,
    "青绿": 2,
    "乌黑": 3,
    "蜷缩": 1,
    "稍蜷": 2,
    "硬挺": 3,
    "沉闷": 1,
    "浊响": 2,
    "清脆": 3,
    "模糊": 1,
    "稍糊": 2,
    "清晰": 3,
    "凹陷": 1,
    "稍凹": 2,
    "平坦": 3,
    "硬滑": 1,
    "软粘": 2,
    "否": 0,
    "是": 1,
}

for k in range(data.shape[0]):
    for i in range(data.shape[1]):
        if data[k, i] in value_dict:
            data[k, i] = value_dict[data[k, i]]

Y = np.array(data[:, -1], dtype=int)

X = np.array(data[:, :-1], dtype=float)
# 样本数、输入层维度
m, d = X.shape
# 隐层维度
q = d + 1
# 输出层维度
l = 1
# 容错空间
eps = 1e-6
# 输出层阈值
ini_theta = np.random.random(l)
# 隐层阈值
ini_gamma = np.random.random(q)
# 输入层和隐层连接权
ini_v = np.random.randn(d, q)
# 隐层和输出层连接权
ini_w = np.random.randn(q, l)
# 超参数
eta = 2.4
df = pd.DataFrame()
Y = Y.reshape((m, l))
while eta <= 3.3:
    errors = []
    theta = ini_theta.copy()
    gamma = ini_gamma.copy()
    v = ini_v.copy()
    w = ini_w.copy()
    print(eta)
    for _ in range(2500):
        error = E(X, Y, theta, gamma, v, w)
        errors.append(error)
        alpha = np.dot(X, v)
        b = sigmoid(alpha - gamma)
        beta = np.dot(b, w)
        y_hat = sigmoid(beta - theta)

        g = y_hat * (1 - y_hat) * (Y - y_hat)
        e = b * (1 - b) * np.dot(g, w.T)
        w += eta * np.dot(b.T, g) / m
        theta -= eta * np.sum(g, axis=0) / m
        v += eta * np.dot(X.T, e) / m
        gamma -= eta * np.sum(e, axis=0) / m
    df["η = {}".format(round(eta, 1))] = errors
    eta += 0.2

sns.set(style="darkgrid")
sns.lineplot(data=[df["η = 2.4"], df["η = 2.6"], df["η = 2.8"], df["η = 3.0"], df["η = 3.2"]])
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
plt.xlabel('迭代次数', fontsize=10)
plt.ylabel('均方误差', fontsize=10)
plt.show()
