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
    for j in range(data.shape[1]):
        if data[k, j] in value_dict:
            data[k, j] = value_dict[data[k, j]]

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
eta = 0.1
df = pd.DataFrame()
while eta <= 1:
    errors = []
    theta = ini_theta.copy()
    gamma = ini_gamma.copy()
    v = ini_v.copy()
    w = ini_w.copy()
    print(eta)
    for _ in range(2500):
        error = E(X, Y, theta, gamma, v, w)
        errors.append(error)
        for k in range(m):
            x = X[k]
            y = Y[k]
            alpha = np.dot(x, v)
            b = sigmoid(alpha - gamma)
            beta = np.dot(b, w)
            y_hat = sigmoid(beta - theta)
            g = y_hat * (1 - y_hat) * (y - y_hat)
            e = b * (1 - b) * np.dot(g, w.T)
            w += eta * (b.reshape((q, 1)) @ g.reshape((1, l)))
            theta -= eta * g
            v += eta * (x.reshape((d, 1)) @ e.reshape((1, q)))
            gamma -= eta * e
    df["η = {}".format(round(eta, 1))] = errors

    eta += 0.2

sns.set(style="darkgrid")
sns.lineplot(data=[df["η = 0.1"], df["η = 0.3"], df["η = 0.5"], df["η = 0.7"], df["η = 0.9"]])
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
plt.xlabel('迭代次数', fontsize=10)
plt.ylabel('均方误差', fontsize=10)
plt.show()
