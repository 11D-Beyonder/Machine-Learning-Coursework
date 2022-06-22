import numpy as np
import pandas as pd


def data_fruit(split_train_test=False, split_sample_target=True):
    dataset = pd.read_csv('Date_Fruit_Datasets/modified.csv').values[:, 1:]
    if split_sample_target:
        X, y = dataset[:, :-1], dataset[:, -1]
        if not split_train_test:
            return X, y
        split = int(len(dataset) * 0.75)
        X_train = np.array(X[:split])
        y_train = y[:split]
        X_test = np.array(X[split:])
        y_test = y[split:]
        return X_train.astype(float), y_train, X_test.astype(float), y_test
    else:
        if not split_train_test:
            return dataset
        else:
            split = int(len(dataset) * 0.75)
            return dataset[:split], dataset[split:]
