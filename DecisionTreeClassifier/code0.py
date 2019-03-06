# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:57:52 2019

@author: WZX
"""
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt

# 鸢尾花数据集
# 建立决策树分类模型 default(gini)
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

# 可视化 to_pdf
dot_date = tree.export_graphviz(clf)
graph = pydotplus.graph_from_dot_data(dot_date)
graph.write_pdf('iris_DecisionTreeClassifier.pdf')

# 可视化
Image(graph.create_jpg())

# 取二维数据(花萼的长)
X = iris.data[:, [0, 2]]
Y = iris.target
clf = tree.DecisionTreeClassifier(max_depth=4)
clf.fit(X, Y)

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 产生两个坐标值矩阵
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))

# ravel将矩阵扁平化为一维向量
# c_矩阵横向拼接 行数不变
# r_矩阵纵向拼接 列数不变
Y_prid = clf.predict(np.c_[x1.ravel(), x2.ravel()])
Y_prid =  Y_prid.reshape(x1.shape)

# 填充等高线图 散点图
plt.figure()
plt.contourf(x1, x2, Y_prid, alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8)

# 取花萼的宽
X = iris.data[:, [1, 3]]
Y = iris.target
clf = tree.DecisionTreeClassifier(max_depth=4)
clf.fit(X, Y)

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))

Y_prid = clf.predict(np.c_[x1.ravel(), x2.ravel()])
Y_prid =  Y_prid.reshape(x1.shape)

plt.figure()
plt.contourf(x1, x2, Y_prid, alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8)
