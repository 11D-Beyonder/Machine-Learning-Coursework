import numpy as np
import matplotlib.pyplot as plt

# [密度,含糖率]
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
# 得到复合矩阵
X = np.c_[X, np.ones(17)]
# 真实标记
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# y=1概率
def p1(x, beta):
    return np.exp(beta @ x.T) / (1 + np.exp(beta @ x.T))


# 一阶偏导
def get_partial1(beta):
    ans = np.zeros(3)
    for i in range(17):
        ans += X[i] * (y[i] - p1(X[i], beta))
    return -ans


# 二阶偏导
def get_partial2(beta):
    ans = np.zeros((3, 3))
    for i in range(17):
        p = p1(X[i], beta)
        ans += (X[i].reshape((3, 1)) @ X[i].reshape((1, 3))) * p * (1 - p)
    return ans


# 目标函数
def ell(beta):
    ans = 0
    for i in range(17):
        ans = ans - y[i] * np.dot(beta, X[i]) + np.log(1 + np.exp(np.dot(beta, X[i])))
    return ans


# 牛顿迭代
def newton(ini_beta, error):
    y = np.zeros(15)
    beta = ini_beta
    old_l = None
    while True:
        subtractor = np.linalg.inv(get_partial2(beta)) @ get_partial1(beta)
        beta = beta - subtractor.reshape((1, 3))[0]
        cur_l = ell(beta)
        if old_l is not None:
            if np.abs(old_l - cur_l) <= error:
                break
        old_l = cur_l
    return beta


Beta = newton(np.ones(3), 1e-6)
print(Beta)
# ========================================= 画图Begin =======================================
plt.figure()
# 原始点
plt.scatter(X[8:, 0], X[8:, 1], color='green', label='bad', alpha=.8, marker='.')
plt.scatter(X[:8, 0], X[:8, 1], color='red', label='good', alpha=.8, marker='.')
# 分割线
plt.plot([0.2, 0.8], [-(Beta[0] * 0.2 + Beta[2]) / Beta[1], -(Beta[0] * 0.8 + Beta[2]) / Beta[1]])
plt.xlabel('density', fontsize=10)
plt.ylabel('sugar content', fontsize=10)
plt.legend()
plt.show()
# ======================================== 画图End ==========================================
