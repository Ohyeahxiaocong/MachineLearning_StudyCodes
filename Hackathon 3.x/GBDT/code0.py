# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:47:39 2019

@author: Hyomin
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# 将训练集分为输入和输出
train = pd.read_csv('datasets/train_modified.csv')
test = pd.read_csv('datasets/test_modified.csv')
x_colums = [x for x in train.columns if x not in ['Disbursed']]
X_train = train[x_colums]
y_train = train['Disbursed']
X_test = test[x_colums]
y_test = test['Disbursed']

# 用默认参数跑模型 auc:0.8702499134014074
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X_train,y_train)
y_pred = gbm0.predict(X_train)
y_predprob = gbm0.predict_proba(X_train)
metrics.roc_auc_score(y_train, y_predprob[:, 1])

# 网格搜索最优迭代次数{'n_estimators': 100}
gbm1 = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1000,
                                  min_samples_leaf=100,max_depth=8, max_features='sqrt',
                                  subsample=0.8, random_state=10)
parameter1 = {'n_estimators': np.arange(30, 150, 10)}
gsearch1 = GridSearchCV(estimator=gbm1, param_grid=parameter1, scoring='roc_auc', iid=False, cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.cv_results_
gsearch1.best_params_

# 树最大深度和最小分类样本点树的最优组合
gbm2 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,
                                  min_samples_leaf=100,max_features='sqrt',
                                  subsample=0.8, random_state=10)
parameter2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 1500, 150)}
gsearch2 = GridSearchCV(estimator=gbm2, param_grid=parameter2, scoring='roc_auc', iid=False, cv=5)
gsearch1.fit(X_train, y_train)