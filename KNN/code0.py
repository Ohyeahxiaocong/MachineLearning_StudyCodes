# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# 形成一个n分类的数据集
# 样本数：1000 特征数：2 冗余：0 每个分类的簇数：1 样本分类数：3
X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1,n_classes=3)

plt.figure()
plt.scatter(X[:, 0], X[:,1], c=Y)

# 建立模型
clf = KNeighborsClassifier(n_neighbors=15, weights='distance')
clf.fit(X, Y)

# 将数字或颜色参数转换为RGB或RGBA的模块
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 建立测试空间
# 获得测试空间的分类结果
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# pcolormesh 使用非常规矩形网格创建伪彩色图
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'distance')" )