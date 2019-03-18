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

# 网格搜索最优迭代次数 {'n_estimators': 100} 0.8477608572190466
gbm1 = GradientBoostingClassifier(learning_rate=0.1,
                                  min_samples_split=1000,
                                  min_samples_leaf=100,
                                  max_depth=8,
                                  max_features='sqrt',
                                  subsample=0.8,
                                  random_state=10)
parameter1 = {'n_estimators': np.arange(30, 150, 10)}
gsearch1 = GridSearchCV(estimator=gbm1, param_grid=parameter1, scoring='roc_auc', iid=False, cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.cv_results_
gsearch1.best_params_

# 树最大深度和最小分类样本树的最优组合 {'max_depth': 7, 'min_samples_split': 1000} 0.8492557310365922
gbm2 = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=100,
                                  min_samples_leaf=100,
                                  max_features='sqrt',
                                  subsample=0.8,
                                  random_state=10)
parameter2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 1500, 150)}
gsearch2 = GridSearchCV(estimator=gbm2, param_grid=parameter2, scoring='roc_auc', iid=False, cv=5)
gsearch2.fit(X_train, y_train)
gsearch2.best_score_
gsearch2.best_params_

# 最小分类样本数和最小叶节点样本数的最优组合 {'min_samples_leaf': 80, 'min_samples_split': 1300} 0.849653033147489
gbm3 = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=7,
                                  max_features='sqrt',
                                  subsample=0.8,
                                  random_state=10)
parameter3 = {'min_samples_leaf': range(60, 130, 20), 'min_samples_split': range(100, 1500, 150)}
gsearch3 = GridSearchCV(estimator=gbm3, param_grid=parameter3, scoring='roc_auc', iid=False, cv=5)
gsearch3.fit(X_train, y_train)
gsearch3.best_score_
gsearch3.best_params_

# 最大特征数调优 {'max_features': 25} 0.8498905820022624
gbm4 = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=7,
                                  min_samples_leaf=80,
                                  min_samples_split=1300,
                                  subsample=0.8,
                                  random_state=10)
parameter4 = {'max_features': range(10, 31, 5)}
gsearch4 = GridSearchCV(estimator=gbm4, param_grid=parameter4, scoring='roc_auc', iid=False, cv=5)
gsearch4.fit(X_train, y_train)
gsearch4.best_score_
gsearch4.best_params_

# 采样率调优 {'subsample': 0.85} 0.8507789563605644
gbm5 = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=7,
                                  min_samples_leaf=80,
                                  min_samples_split=1300,
                                  max_features=25,
                                  random_state=10)
parameter5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
gsearch5 = GridSearchCV(estimator=gbm5, param_grid=parameter5, scoring='roc_auc', iid=False, cv=5)
gsearch5.best_score_
gsearch5.best_params_

# 验证调优后的参数 auc:0.906034684455213
gbm6 = GradientBoostingClassifier(learning_rate=0.1,
                                  n_estimators=100,
                                  max_depth=7,
                                  min_samples_leaf=80,
                                  min_samples_split=1300,
                                  max_features=25,
                                  subsample=0.85,
                                  random_state=10)
gbm6.fit(X_train,y_train)
y_pred = gbm6.predict(X_train)
y_predprob = gbm6.predict_proba(X_train)
metrics.accuracy_score(y_train,y_pred)
metrics.roc_auc_score(y_train, y_predprob[:, 1])