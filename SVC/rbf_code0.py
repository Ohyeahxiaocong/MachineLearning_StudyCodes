# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 在2d中创建一个包含较小圆的大圆
# 样本点1000个 噪声0.2 内圈和外圈之间的比例因子0.5
X, Y = make_circles(n_samples=100, noise=0.2, factor=0.5)

# 样本点归一化
X = StandardScaler().fit_transform(X)

# 样本点可视化
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
plt.xticks(())
plt.yticks(())
plt.title('Input Data')

# 选取最优超参数组合
grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 0.3, 0.6, 1, 3, 6, 10],
                    "gamma": [1, 0.6, 0.3, 0.1, 0.06, 0.03, 0.01]}, cv=10)
grid.fit(X, Y)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.02),np.arange(y_min, y_max, 0.02))

# 固定gamma
for C in [0.1, 0.3, 0.6, 1, 3, 6, 10]:
    gamma = 0.6
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X, Y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('gamma=' + str(gamma) + 'C=' + str(C))
    
# 固定C
for gamma in [0.1, 0.3, 0.6, 1, 3, 6, 10]:
    C = 0.6
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X, Y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('gamma=' + str(gamma) + 'C=' + str(C))