import numpy as np
import matplotlib.pyplot as plt

# 获取投影坐标
def get_projection(w, X):
    # 到原点距离
    dis = w.T @ X.T / np.linalg.norm(w)

    # 夹角
    theta = np.arctan(w[1] / w[0])

    # 返回投影点坐标
    return dis * np.cos(theta), dis * np.sin(theta)


# ============================================ 计算Begin ==============================================
# 正例
X1 = np.array([
    [0.697, 0.460],
    [0.774, 0.376],
    [0.634, 0.264],
    [0.608, 0.318],
    [0.556, 0.215],
    [0.403, 0.237],
    [0.481, 0.149],
    [0.437, 0.211]
])

# 反例
X0 = np.array([
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
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# 均值向量（列向量）
mu0 = np.mean(X0, axis=0).reshape((2, 1))
mu1 = np.mean(X1, axis=0).reshape((2, 1))

# 协方差矩阵
Sigma0 = (X0.T - mu0) @ (X0.T - mu0).T
Sigma1 = (X1.T - mu1) @ (X1.T - mu1).T

# 类内散度矩阵
Sw = Sigma0 + Sigma1

# 直线方向向量
w = np.linalg.inv(Sw) @ (mu0 - mu1)

# 输出直线斜率
print(w[1] / w[0])

# 使直线朝向为y正半轴
if w[1] < 0:
    w = -w

# ============================================ 计算End ==============================================

# ============================================ 画图Begin ============================================

# 画图
plt.figure()

# 原始点
plt.scatter(X0[:, 0], X0[:, 1], color='green', label='bad', alpha=.8, marker='.')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='good', alpha=.8, marker='.')

# 坏瓜投影点
pro_X, pro_Y = get_projection(w, X0)
plt.scatter(pro_X, pro_Y, color='green', label='bad(projection)', alpha=.8, marker='x')
# 垂直线
for i in range(9):
    plt.plot([pro_X.T[i], X0[i][0]], [pro_Y.T[i][0], X0[i][1]], color='green', linestyle='--', linewidth=0.5)

# 坏瓜投影点
pro_X, pro_Y = get_projection(w, X1)
plt.scatter(pro_X, pro_Y, color='red', label='good(projection)', alpha=.8, marker='x')
# 垂直线syntax
for i in range(8):
    plt.plot([pro_X.T[i], X1[i][0]], [pro_Y.T[i][0], X1[i][1]], color='red', linestyle='--', linewidth=0.5)

# 投影直线
plt.plot([0, w[0]], [0, w[1]], label=r'$y=w^T$x')

plt.xlabel('density', fontsize=10)
plt.ylabel('sugar content', fontsize=10)
plt.legend()
plt.show()

# ============================================ 画图End ============================================
