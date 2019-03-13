# -*- coding: utf-8 -*-
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

# 高斯分布
clf = GaussianNB()

# 拟合数据集
clf.fit(X, Y)

# 分批拟合数据集
clf.partial_fit(X, Y)

# 返回预测结果
clf.predict([[-0.8, -1]])

# 返回每个分类的预测概率
clf.predict_proba([[-0.8, -1]])

# 返回每个分类预测的概率的一个对数转化
clf.predict_log_proba([[-0.8, -1]])