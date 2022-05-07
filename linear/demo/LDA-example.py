import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

# 数据集
X = iris.data
# 结果集
y = iris.target
target_names = iris.target_names
# n_components分类数-1（用于降维）
lda = LinearDiscriminantAnalysis(n_components=2)
# fit拟合模型
# transform 将数据投射到最大分类块中
res = lda.fit(X, y).transform(X)

colors = ['navy', 'turquoise', 'darkorange']
plt.figure()
# 遍历每个类别
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(res[y == i, 0], res[y == i, 1], alpha=.8, color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()
