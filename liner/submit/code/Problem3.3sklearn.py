from sklearn.linear_model import LogisticRegression
import numpy as np

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
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
clf = LogisticRegression(solver='newton-cg').fit(X, y)
# 输出w和b
print(np.c_[clf.coef_, clf.intercept_])
