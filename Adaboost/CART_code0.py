# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:38:47 2019

@author: WZX
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# 高斯分布随机数
X1, Y1 = make_gaussian_quantiles(cov=2.0, n_samples=500, n_features=2,
                                 n_classes=2, random_state=1)
X2, Y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=400,
                                 n_features=2, n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
Y = np.concatenate((Y1, - Y2 + 1))

# 采用了CART树 最大树深为2 内部节点再划分所需最小样本数20 叶子节点最少样本数5
# 迭代次数200次 学习率0.8 (组合考虑)
# SAMME.R 需要返回预测每个分类标签的概率
# SAMME 返回预测概率最大的标签值
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20,
    min_samples_leaf=5), algorithm="SAMME", n_estimators=200, learning_rate=0.8)
bdt.fit(X, Y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], marker='.', c=Y)