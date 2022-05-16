"""
以决策树桩为基学习器
训练Bagging集成
"""
import numpy as np


def cal_ent():

def cal_gain(feature_index, D):
    feature_values = X[:, feature_index]
    sorted_feature_values = np.sort(feature_values)
    split_points = [(sorted_feature_values[i] + sorted_feature_values[i + 1]) / 2 for i in range(m - 1)]
    for i in range(m - 1):


def build_best_stump(D):
    best_split_point = None
    min_error = np.inf
    best_sign = None
    best_feature = None
    for feature_index in range(n):
        split_point, sign, error = get_best_split_point(feature_index, D)
        if error < min_error:
            min_error = error
            best_feature = feature_index
            best_split_point = split_point
            best_sign = sign
    return best_feature, best_split_point, best_sign, min_error


if __name__ == '__main__':
    data = pd.read_csv('watermelon3.0a.csv', header=0).values
    X = data[:, :-1]
    m, n = X.shape
    y = data[:, -1]
