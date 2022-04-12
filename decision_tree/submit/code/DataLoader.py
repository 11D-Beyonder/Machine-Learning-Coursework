import numpy as np
import pandas as pd

"""
数据读取
"""


# tic_tac_toe = pd.read_csv('../data/tic-tac-toe.data', header=None)
# X = tic_tac_toe.iloc[:, :9].values
# y = tic_tac_toe.iloc[:, -1].values
# D = np.c_[X, y]
# np.random.shuffle(D)
# pd.DataFrame(D, columns=None).to_csv('../data/modify.csv')


def load(data_name):
    if data_name == 'lymphography':
        return load_lymphography()
    elif data_name == 'balance-scale':
        return load_balance_scale()
    elif data_name == 'tic-tac-toe':
        return load_tic_tac_toe()
    elif data_name == 'ionosphere':
        return load_ionosphere()
    elif data_name == 'watermelon2.0':
        return load_watermelon2()
    else:
        raise Exception('数据源不存在')


def load_lymphography():
    """
    读取lymphography.data
    第1列为分类，其余为属性值。
    :return: 返回属性值，真实标记，每个属性可取的值。
    """
    lymphography = pd.read_csv('../data/lymphography.data', header=None)
    X = lymphography.iloc[:, 1::].values
    y = lymphography.iloc[:, 0].values
    class_dicts = {}
    for i in range(18):
        class_dicts[i] = set(X[:, i])
    D = np.c_[X, y]
    return D, class_dicts


def load_balance_scale():
    """
    读取balance_scale.data
    第1列为分类，其余为属性值。
    :return: 属性值，真实标记，每个属性可取的值。
    """
    balance_scale = pd.read_csv('../data/balance-scale.data', header=None)
    X = balance_scale.iloc[:, 1::].values
    y = balance_scale.iloc[:, 0].values
    class_dicts = {}
    for i in range(4):
        class_dicts[i] = set(X[:, i])
    D = np.c_[X, y]
    return D, class_dicts


def load_tic_tac_toe():
    """
    读取tic-tac-toe.data
    最后一列为分类，其余为属性值
    :return: 属性值，真实标记，每个属性可取的值。
    """
    tic_tac_toe = pd.read_csv('../data/modify.csv', header=0)
    X = tic_tac_toe.iloc[:, 1:10].values
    y = tic_tac_toe.iloc[:, -1].values
    D = np.c_[X, y]
    class_dicts = {}
    for i in range(9):
        class_dicts[i] = set(X[:, i])
    return D, class_dicts


def load_ionosphere():
    """
    读取ionosphere.data
    最后一列为分类，其余为属性值
    :return: 属性值，真实标记
    """
    ionosphere = pd.read_csv('../data/ionosphere.data', header=None)
    X = ionosphere.iloc[:, :-1].values
    y = ionosphere.iloc[:, -1].values
    D = np.c_[X, y]
    return D


def load_watermelon2():
    """
    读取watermelon2.0.data
    最后一列为分类，其余为属性值
    :return: 属性值，真实标记
    """
    ionosphere = pd.read_csv('../data/watermelon2.0.data', header=0)
    X = ionosphere.iloc[:, 1:-1].values
    y = ionosphere.iloc[:, -1].values
    D = np.c_[X, y]
    return D
