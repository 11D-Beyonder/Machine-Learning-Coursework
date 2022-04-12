import numpy as np
import collections


def split_data_and_target(D):
    """
    :param D: 训练集
    :return: 拆分为参数和真实值
    """
    if len(D) == 0:
        return [], []
    else:
        return D[:, :-1], D[:, -1]


def Ent(D):
    """
    计算信息熵
    :param D: 训练集
    :return: 信息熵的值
    """
    ans = 0
    X, y = split_data_and_target(D)
    counter = collections.Counter(y)
    # 共有tot个样本
    tot = len(y)
    # 遍历不同样本各自存在的个数
    for attribute, count in counter.items():
        # 计算所占比例
        p = count / tot
        ans -= p * np.log2(p)
    return ans


def get_Dv(D, a, v):
    """
    得到子集 Dv
    :param D: 训练集
    :param a: 属性索引
    :param v: 属性值
    :return: 属性a上取值为v的子集Dv
    """
    Dv = []
    for d in D:
        if d[a] == v:
            Dv.append(d)
    return np.array(Dv)


def Gain(D, a, class_dicts):
    """
    :param D: 训练集
    :param a: 属性索引
    :param class_dicts: 属性可取值
    :return: 信息增益
    """
    ans = Ent(D)
    # 遍历属性a的所有可能取值
    for v in class_dicts[a]:
        # 找到属性a取值为v的样本
        Dv = get_Dv(D, a, v)
        ans -= len(Dv) / len(D) * Ent(Dv)
    return ans


def Gini(D):
    """
    :param D: 训练集
    :return: 基尼值
    """
    ans = 0
    X, y = split_data_and_target(D)
    counter = collections.Counter(y)
    # 共有tot个样本
    tot = len(y)
    # 遍历不同样本各自存在的个数
    for attribute, count in counter.items():
        # 计算所占比例
        p = count / tot
        ans += p * p
    ans = 1 - ans
    return ans


def Gini_index(D, a, class_dicts):
    """
    :param D: 训练集
    :param a: 属性索引
    :param class_dicts: 属性可取的值
    :return: 基尼系数
    """
    ans = 0
    for v in class_dicts[a]:
        # 找到属性a取值为v的样本
        Dv = get_Dv(D, a, v)
        if len(Dv) == 0:
            continue
        ans += len(Dv) / len(D) * Gini(Dv)
    return ans


def count_value_num(X, v):
    """
    :param X: 数组X
    :param v: 值x
    :return: X中取值为v的个数
    """
    ans = 0
    for x in X:
        if all(x == v):
            ans += 1
    return ans
