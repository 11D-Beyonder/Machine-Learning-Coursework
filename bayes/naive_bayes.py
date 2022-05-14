import numpy as np
import pandas as pd

df = pd.read_csv('watermelon3.csv', header=0).iloc[:, 1:]
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
N = len(X)
d = len(X[0])


def get_P(c, xi=None, i=None):
    if xi is not None and i is not None:
        if df.columns.values[i] == '含糖率' or df.columns.values[i] == '密度':
            mu = 0
            num = 0
            for k in range(N):
                if y[k] == c:
                    num += 1
                    mu += X[k][i]
            mu /= num
            sigma = 0
            for k in range(N):
                if y[k] == c:
                    sigma += (mu - X[k][i]) ** 2
            sigma /= (num - 1)
            sigma = np.sqrt(sigma)
            return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(xi - mu) ** 2 / (2 * sigma ** 2))
        else:
            # 类别为c第i个属性上取值为xi的个数
            num1 = 0
            # 类别为c的个数
            num2 = 0
            for k in range(N):
                if y[k] == c:
                    num2 += 1
                    if xi == X[k][i]:
                        num1 += 1

            return (num1 + 1) / (num2 + len(set(X[:, i])))
    else:
        num = 0
        for k in range(N):
            if y[k] == c:
                num += 1
        return (num + 1) / (N + 2)


def predict(ex):
    max_value = 0
    ans = ''
    for c in ['是', '否']:
        nb = 1
        for i in range(d):
            nb *= get_P(c, ex[i], i)
        nb *= get_P(c)
        print('P({})={}'.format(c, nb))
        if nb > max_value:
            max_value = nb
            ans = c
    return ans


print(predict(['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.46]))
