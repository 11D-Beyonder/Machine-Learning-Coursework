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


def get_error(X, Y, theta, gamma, v, w):
    ans = 0
    for k in range(m):
        x = X[k]
        y = Y[k]
        y_hat = predict(x, theta, gamma, v, w)
        ans += np.linalg.norm(y - y_hat) ** 2
    return ans / m


data = np.array(pd.read_csv("../data/watermelon3.0.csv").iloc[:, 1:])

value_dict = {
    '浅白': 1, '青绿': 2, '乌黑': 3,
    '蜷缩': 1, '稍蜷': 2, '硬挺': 3,
    '沉闷': 1, '浊响': 2, '清脆': 3,
    '模糊': 1, '稍糊': 2, '清晰': 3,
    '凹陷': 1, '稍凹': 2, '平坦': 3,
    '硬滑': 1, '软粘': 2,
    '否': 0, '是': 1
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
theta = np.random.random(l)
# 隐层阈值
gamma = np.random.random(q)
# 输入层和隐层连接权
V = np.random.randn(d, q)
# 隐层和输出层连接权
W = np.random.randn(q, l)
# 超参数
eta = 2

errors = []
Y = Y.reshape((m, l))
while True:
    # 计算均方误差
    error = get_error(X, Y, theta, gamma, V, W)
    errors.append(error)
    if len(errors) > 1 and np.abs(error - errors[len(errors) - 2]) <= eps:
        break
    # 得到预测值
    Alpha = np.dot(X, V)
    B = sigmoid(Alpha - gamma)
    Beta = np.dot(B, W)
    Y_hat = sigmoid(Beta - theta)
    # 更新参数
    G = Y_hat * (1 - Y_hat) * (Y - Y_hat)
    E = B * (1 - B) * np.dot(G, W.T)
    W += eta * np.dot(B.T, G) / m
    theta -= eta * np.sum(G, axis=0) / m
    V += eta * np.dot(X.T, E) / m
    gamma -= eta * np.sum(E, axis=0) / m

sns.set(style='darkgrid')
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
sns.lineplot(x='time', y='errors',
             data=pd.DataFrame({'time': np.arange(len(errors)), 'errors': errors}), label='$\eta={}$'.format(eta))
plt.xlabel('迭代次数', fontsize=10)
plt.ylabel('均方误差', fontsize=10)
plt.show()
