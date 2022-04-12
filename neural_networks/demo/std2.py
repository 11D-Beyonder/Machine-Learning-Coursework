import numpy as np
import pandas as pd

data = np.array(pd.read_csv("../submit/data/watermelon3.0.csv").iloc[:, 1:])

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

trueY = np.array(data[:, -1], dtype=int)
X = np.array(data[:, :-1], dtype=float)
m, n = np.shape(X)

# according to P101, init the parameters
import random

d = n  # the dimension of the input vector
l = 1  # the dimension of the  output vector
q = d + 1  # the number of the hide nodes
theta = [random.random() for i in range(l)]  # the threshold of the output nodes
gamma = [random.random() for i in range(q)]  # the threshold of the hide nodes
# v size= d*q .the connection weight between input and hide nodes
v = [[random.random() for i in range(q)] for j in range(d)]
# w size= q*l .the connection weight between hide and output nodes
w = [[random.random() for i in range(l)] for j in range(q)]
eta = 0.2  # the training speed
maxIter = 5000  # the max training times

import math


def sigmoid(iX, dimension):  # iX is a matrix with a dimension
    if dimension == 1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = sigmoid(iX[i], dimension - 1)
    return iX


# do the repeat----standard BP
# while maxIter > 0:
#     maxIter -= 1
#     sumE = 0
#     for i in range(m):
#         alpha = np.dot(X[i], v)  # p101 line 2 from bottom, shape=1*q
#         b = sigmoid(alpha - gamma, 1)  # b=f(alpha-gamma), shape=1*q
#         beta = np.dot(b, w)  # shape=(1*q)*(q*l)=1*l
#         predictY = sigmoid(beta - theta, 1)  # shape=1*l ,p102--5.3
#         E = sum((predictY - trueY[i]) * (predictY - trueY[i])) / 2  # 5.4
#         sumE += E  # 5.16
#         # p104
#         g = predictY * (1 - predictY) * (trueY[i] - predictY)  # shape=1*l p103--5.10
#         e = b * (1 - b) * ((np.dot(w, g.T)).T)  # shape=1*q , p104--5.15
#         w += eta * np.dot(b.reshape((q, 1)), g.reshape((1, l)))  # 5.11
#         theta -= eta * g  # 5.12
#         v += eta * np.dot(X[i].reshape((d, 1)), e.reshape((1, q)))  # 5.13
#         gamma -= eta * e  # 5.14
# print(sumE)


# accumulated BP
trueY = trueY.reshape((m, l))
while maxIter > 0:
    maxIter -= 1
    sumE = 0
    alpha = np.dot(X, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid(alpha - gamma, 2)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid(beta - theta, 2)  # shape=m*l ,p102--5.3

    E = sum(sum((predictY - trueY) * (predictY - trueY))) / 2  # 5.4
    # print(round(E,5))
    g = predictY * (1 - predictY) * (trueY - predictY)  # shape=m*l p103--5.10
    e = b * (1 - b) * ((np.dot(w, g.T)).T)  # shape=m*q , p104--5.15
    w += eta * np.dot(b.T, g)  # 5.11 shape (q*l)=(q*m) * (m*l)
    theta -= eta * g  # 5.12
    v += eta * np.dot(X.T, e)  # 5.13 (d,q)=(d,m)*(m,q)
    gamma -= eta * e  # 5.14


def predict(iX):
    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid(alpha - gamma, 2)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid(beta - theta, 2)  # shape=m*l ,p102--5.3
    return predictY


print(predict(X))
