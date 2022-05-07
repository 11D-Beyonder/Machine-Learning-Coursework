import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
max_iter = 50
m = 17
X = np.array([
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211],
    [0.666, 0.091],
    [0.243, 0.267],
    [0.245, 0.057],
    [0.343, 0.099],
    [0.639, 0.161],
    [0.657, 0.198],
    [0.360, 0.370],
    [0.593, 0.042],
    [0.719, 0.103]
])
# 真实标记
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
# 初始化
alpha = np.zeros(m)
b = 0
# 常数取值
epsilon = 0.00001
C = 1


def f(_x):
    ans = b
    for i in range(m):
        ans += alpha[i] * y[i] * kappa(_x, X[i])
    return ans


def kappa(x1, x2, func='gauss'):
    if func == 'gauss':
        sigma = 0.11855
        return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * sigma ** 2))
    elif func == 'linear':
        return np.dot(x1, x2)
    elif func == 'polynomial':
        d = 5
        return np.dot(x1, x2) ** d
    elif func == 'laplace':
        sigma = 0.11855
        return np.exp(-np.linalg.norm(x1 - x2) / sigma)
    elif func == 'sigmoid':
        beta = 0.8
        theta = -0.2
        return np.tanh(beta * np.dot(x1, x2) + theta)
    elif func == 'RBF':
        gamma = 0.2
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    else:
        raise Exception('核函数不存在')


def clip(a, _L, _H):
    if a < _L:
        return _L
    elif a > _H:
        return _H
    else:
        return a


for _ in range(max_iter):
    index1 = index2 = -1
    # 寻找不满足约束的alpha_i
    for i in range(m):
        Ei = f(X[i]) - y[i]
        if alpha[i] < C and y[i] * Ei < -epsilon:
            index1 = i
            break
        if alpha[i] > 0 and y[i] * Ei > epsilon:
            index1 = i
            break

    # 都满足约束则随机选择
    if index1 == -1:
        index1 = np.random.randint(0, m)

    E1 = f(X[index1]) - y[index1]

    # 寻找使更新步长最大的alpha_j
    E2 = E1
    for i in range(m):
        if i == index1:
            continue
        Ei = f(X[i]) - y[i]
        if np.abs(Ei - E1) > np.abs(E2 - E1):
            E2 = Ei
            index2 = i

    alpha1_old = alpha[index1]
    alpha2_old = alpha[index2]
    # 更新
    alpha2_new = alpha2_old + y[index2] * (E1 - E2) / (
            kappa(X[index1], X[index1]) + kappa(X[index2], X[index2]) - 2 * kappa(X[index1], X[index2]))
    # 确定可行域
    if y[index1] == y[index2]:
        L = max(0, alpha1_old + alpha2_old - C)
        H = min(C, alpha2_old + alpha1_old)
    else:
        L = max(0, alpha2_old - alpha1_old)
        H = min(C, alpha2_old - alpha1_old + C)

    # 修剪
    alpha2_new = clip(alpha2_new, L, H)

    alpha1_new = alpha1_old + y[index1] * y[index2] * (alpha2_old - alpha2_new)
    alpha[index1] = alpha1_new
    alpha[index2] = alpha2_new

    # 更新b
    b1 = -E1 - y[index1] * kappa(X[index1], X[index1]) * (alpha1_new - alpha1_old) - y[index2] * kappa(X[index1],
                                                                                                       X[index2]) * (
                 alpha2_new - alpha2_old) + b
    b2 = -E2 - y[index1] * kappa(X[index1], X[index2]) * (alpha1_new - alpha1_old) - y[index2] * kappa(X[index2],
                                                                                                       X[index2]) * (
                 alpha2_new - alpha2_old) + b
    if 0 < alpha1_new < C:
        b = b1
    elif 0 < alpha2_new < C:
        b = b2
    elif L != H:
        b = (b1 + b2) / 2

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

all = np.c_[xx.ravel(), yy.ravel()]
Z = np.empty(len(all))
for i in range(len(all)):
    Z[i] = f(all[i])
    if Z[i] < 0:
        Z[i] = -1
    else:
        Z[i] = 1
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)

plt.scatter(X[:8, 0], X[:8, 1], c='orange', label='好瓜',
            s=20, edgecolor='k')
plt.scatter(X[8:, 0], X[8:, 1], c='blue', label='坏瓜',
            s=20, edgecolor='k')
plt.xlabel('密度')
plt.ylabel('含糖率')
plt.legend()
plt.show()
