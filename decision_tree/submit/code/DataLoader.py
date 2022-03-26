import numpy as np
import pandas as pd

"""
数据读取
"""


def load(data_num):
    if data_num == 'lymphography':
        return load_lymphography()
    elif data_num == 'balance-scale':
        return load_balance_scale()


def load_lymphography():
    """
    读取lymphography.data
    第1列为分类，其余为属性值。
    返回属性值，真实标记，每个属性可取的值。
    """
    lymphography = pd.read_csv('../data/lymphography.data', header=None)
    X = lymphography.iloc[:, 1::].values
    y = lymphography.iloc[:, 0].values
    class_dicts = {}
    for i in range(18):
        class_dicts[i] = set(X[:, i])
    return np.c_[X, y], class_dicts


def load_balance_scale():
    """
    读取balance_scale.data
    第1列为分类，其余为属性值。
    返回属性值，真实标记，每个属性可取的值。
    """
    balance_scale = pd.read_csv('../data/balance-scale.data', header=None)
    X = balance_scale.iloc[:, 1::].values
    y = balance_scale.iloc[:, 0].values
    class_dicts = {}
    for i in range(4):
        class_dicts[i] = set(X[:, i])
    return np.c_[X, y], class_dicts
