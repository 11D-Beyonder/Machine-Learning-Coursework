from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut

# 取前100个数据
iris = datasets.load_iris()
x = iris.data[50:150, :]
y = iris.target[50:150]

logisticRegression = LogisticRegression()

# 10折交叉验证
y_predict = cross_val_predict(logisticRegression, x, y, cv=10)
print("10折交叉验证错误率：", 1 - metrics.accuracy_score(y, y_predict))

# 留一法
leaveOneOut = LeaveOneOut()
# 命中数
hits = 0
# 分割为训练集和测试集
for train, test in leaveOneOut.split(x):
    logisticRegression.fit(x[train], y[train])
    p = logisticRegression.predict(x[test])
    if p == y[test]:
        hits += 1
print("留一法错误率", 1 - hits / 100)


