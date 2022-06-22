import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt

import data_loader

X_train, y_train, X_test, y_test = data_loader.data_fruit(split_train_test=True, split_sample_target=True)
target_names = ['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY']
y_train_label = np.empty(len(y_train))
y_test_label = np.empty(len(y_test))
for i in range(len(y_train)):
    y_train_label[i] = target_names.index(y_train[i])
for i in range(len(y_test)):
    y_test_label[i] = target_names.index(y_test[i])
feature_names = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY', 'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA',
                 'EXTENT', 'ASPECT_RATIO', 'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
                 'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'MeanRR', 'MeanRG', 'MeanRB', 'StdDevRR', 'StdDevRG', 'StdDevRB',
                 'SkewRR', 'SkewRG', 'SkewRB', 'KurtosisRR', 'KurtosisRG', 'KurtosisRB', 'EntropyRR', 'EntropyRG',
                 'EntropyRB', 'ALLdaub4RR', 'ALLdaub4RG', 'ALLdaub4RB']
# https://blog.csdn.net/oppo62258801/article/details/82883806
dtrain = xgb.DMatrix(X_train, label=y_train_label, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test_label, feature_names=feature_names)
bst = xgb.train(params={'max_depth': 2, 'objective': 'multi:softmax', 'num_class': 7}, dtrain=dtrain)
y_predict = np.around(bst.predict(dtest))
accuracy = 0
for i in range(len(y_test)):
    if y_predict[i] == y_test_label[i]:
        accuracy += 1

print(accuracy / len(y_test))

for i in range(int(bst.attributes()['best_ntree_limit'])):
    xgb.to_graphviz(bst, num_trees=i).render('image/xgb_tree-{}.gv'.format(i))
xgb.plot_importance(bst)
plt.tight_layout()
plt.show()
