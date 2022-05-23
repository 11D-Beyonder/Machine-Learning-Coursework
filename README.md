在《机器学习》课程学习中留下的代码和笔记。

主要实现了

- 线性模型
    - Logistic Regression
    - LDA
- 决策树
    - 未剪枝
    - 预剪枝
    - 后剪枝
    - 基于信息增益的决策树
    - 基于基尼系数的决策树
    - 多变量决策树（结点上进行Logistic回归）
    - 使用graphviz得到决策树图像
    - 比较了不同
- 神经网络
  - 标准BP算法
  - 累积BP算法
  - 比较了两个算法的收敛速率
  - 比较了不同学习率时均方误差的抖动情况
- SVM
  - 使用SMO算法实现了带软间隔的SVM
  - 比较了不同核函数的分类效果
- 贝叶斯分类器
  - 拉普拉斯修正的朴素贝叶斯分类器
- 集成学习
  - 基于决策树桩的AdaBoost
  - 基于限制深度决策树的AdaBoost
  - 基于1NN的Bagging
  - 基于决策树桩的Bagging