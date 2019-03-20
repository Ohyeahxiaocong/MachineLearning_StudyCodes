# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:47:39 2019

@author: Hyomin
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split

# 将训练集分为输入和输出
dataset = pd.read_csv('datasets/train_modified.csv')
x_colums = [x for x in dataset.columns if x not in ['Disbursed']]
X = dataset[x_colums]
y = dataset['Disbursed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 用默认参数跑模型
#train_accuracy:0.985305
#test_accuracy:0.985635
#train_auc:0.886390
#test_auc:0.842154
gbm0 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=100,
                                  subsample=0.8,
                                  min_samples_split=500,
                                  min_samples_leaf=50,
                                  max_depth=5,
                                  max_features='sqrt',
                                  random_state=10)
gbm0.fit(X_train,y_train)

y_train_pred = gbm0.predict(X_train)
y_test_pred = gbm0.predict(X_test)
y_train_predprob = gbm0.predict_proba(X_train)
y_test_predprob = gbm0.predict_proba(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_auc = metrics.roc_auc_score(y_train, y_train_predprob[:, 1])
test_auc = metrics.roc_auc_score(y_test, y_test_predprob[:, 1])
print('train_accuracy:%.6f' % train_accuracy)
print('test_accuracy:%.6f' % test_accuracy)
print('train_auc:%.6f' % train_auc)
print('test_auc:%.6f' % test_auc)

# 网格搜索最优迭代次数 {'n_estimators': 140} 
#train_accuracy:0.985319
#test_accuracy:0.985635
#train_auc:0.893536
#test_auc:0.843665
gbm1 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=100,
                                  subsample=0.8,
                                  min_samples_split=500,
                                  min_samples_leaf=50,
                                  max_depth=5,
                                  max_features='sqrt',
                                  random_state=10)
parameter1 = {'n_estimators': np.arange(120, 200, 10)}
gsearch1 = GridSearchCV(estimator=gbm1, param_grid=parameter1, scoring='roc_auc',
                        n_jobs=4, iid=False, cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.best_params_

y_train_pred = gsearch1.predict(X_train)
y_test_pred = gsearch1.predict(X_test)
y_train_predprob = gsearch1.predict_proba(X_train)
y_test_predprob = gsearch1.predict_proba(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_auc = metrics.roc_auc_score(y_train, y_train_predprob[:, 1])
test_auc = metrics.roc_auc_score(y_test, y_test_predprob[:, 1])
print('train_accuracy:%.6f' % train_accuracy)
print('test_accuracy:%.6f' % test_accuracy)
print('train_auc:%.6f' % train_auc)
print('test_auc:%.6f' % test_auc)

# 树最大深度和最小分类样本树的最优组合 {'max_depth': 7, 'min_samples_split': 900}
#train_accuracy:0.985334
#test_accuracy:0.985635
#train_auc:0.912968
#test_auc:0.846042
gbm2 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=140,
                                  subsample=0.8,
                                  min_samples_split=500,
                                  min_samples_leaf=50,
                                  max_depth=5,
                                  max_features='sqrt',
                                  random_state=10)
parameter2 = {'max_depth': range(3, 11, 2), 'min_samples_split': range(100, 1300, 200)}
gsearch2 = GridSearchCV(estimator=gbm2, param_grid=parameter2, scoring='roc_auc',
                        n_jobs=4, iid=False, cv=5)
gsearch2.fit(X_train, y_train)
gsearch2.best_params_

y_train_pred = gsearch2.predict(X_train)
y_test_pred = gsearch2.predict(X_test)
y_train_predprob = gsearch2.predict_proba(X_train)
y_test_predprob = gsearch2.predict_proba(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_auc = metrics.roc_auc_score(y_train, y_train_predprob[:, 1])
test_auc = metrics.roc_auc_score(y_test, y_test_predprob[:, 1])
print('train_accuracy:%.6f' % train_accuracy)
print('test_accuracy:%.6f' % test_accuracy)
print('train_auc:%.6f' % train_auc)
print('test_auc:%.6f' % test_auc)

# 最小分类样本数和最小叶节点样本数的最优组合 {'min_samples_leaf': 50}
#train_accuracy:0.985334
#test_accuracy:0.985635
#train_auc:0.912968
#test_auc:0.846042
gbm3 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=140,
                                  subsample=0.8,
                                  min_samples_split=900,
                                  min_samples_leaf=50,
                                  max_depth=7,
                                  max_features='sqrt',
                                  random_state=10)
parameter3 = {'min_samples_leaf': range(30, 120, 20)}
gsearch3 = GridSearchCV(estimator=gbm3, param_grid=parameter3, scoring='roc_auc',
                        n_jobs=4, iid=False, cv=5)
gsearch3.fit(X_train, y_train)
gsearch3.best_score_
gsearch3.best_params_

y_train_pred = gsearch3.predict(X_train)
y_test_pred = gsearch3.predict(X_test)
y_train_predprob = gsearch3.predict_proba(X_train)
y_test_predprob = gsearch3.predict_proba(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_auc = metrics.roc_auc_score(y_train, y_train_predprob[:, 1])
test_auc = metrics.roc_auc_score(y_test, y_test_predprob[:, 1])
print('train_accuracy:%.6f' % train_accuracy)
print('test_accuracy:%.6f' % test_accuracy)
print('train_auc:%.6f' % train_auc)
print('test_auc:%.6f' % test_auc)

# 最大特征数调优 {'max_features': 10}
#train_accuracy:0.985363
#test_accuracy:0.985635
#train_auc:0.917098
#test_auc:0.847754
gbm4 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=140,
                                  subsample=0.8,
                                  min_samples_split=900,
                                  min_samples_leaf=50,
                                  max_depth=7,
                                  max_features='sqrt',
                                  random_state=10)
parameter4 = {'max_features': range(10, 31, 5)}
gsearch4 = GridSearchCV(estimator=gbm4, param_grid=parameter4, scoring='roc_auc',
                        n_jobs=4, iid=False, cv=5)
gsearch4.fit(X_train, y_train)
gsearch4.best_score_
gsearch4.best_params_

y_train_pred = gsearch4.predict(X_train)
y_test_pred = gsearch4.predict(X_test)
y_train_predprob = gsearch4.predict_proba(X_train)
y_test_predprob = gsearch4.predict_proba(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_auc = metrics.roc_auc_score(y_train, y_train_predprob[:, 1])
test_auc = metrics.roc_auc_score(y_test, y_test_predprob[:, 1])
print('train_accuracy:%.6f' % train_accuracy)
print('test_accuracy:%.6f' % test_accuracy)
print('train_auc:%.6f' % train_auc)
print('test_auc:%.6f' % test_auc)

# 采样率调优 {'subsample': 0.85}
#train_accuracy:0.985391
#test_accuracy:0.985578
#train_auc:0.916803
#test_auc:0.843284
gbm5 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=140,
                                  subsample=0.8,
                                  min_samples_split=900,
                                  min_samples_leaf=50,
                                  max_depth=7,
                                  max_features=10,
                                  random_state=10)
parameter5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
gsearch5 = GridSearchCV(estimator=gbm5, param_grid=parameter5, scoring='roc_auc',
                        n_jobs=4, iid=False, cv=5)
gsearch5.fit(X_train, y_train)
gsearch5.best_score_
gsearch5.best_params_

y_train_pred = gsearch5.predict(X_train)
y_test_pred = gsearch5.predict(X_test)
y_train_predprob = gsearch5.predict_proba(X_train)
y_test_predprob = gsearch5.predict_proba(X_test)
train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
train_auc = metrics.roc_auc_score(y_train, y_train_predprob[:, 1])
test_auc = metrics.roc_auc_score(y_test, y_test_predprob[:, 1])
print('train_accuracy:%.6f' % train_accuracy)
print('test_accuracy:%.6f' % test_accuracy)
print('train_auc:%.6f' % train_auc)
print('test_auc:%.6f' % test_auc)

# 验证调优后的参数 auc:0.906034684455213
gbm6 = GradientBoostingClassifier(loss='deviance',
                                  learning_rate=0.1,
                                  n_estimators=140,
                                  subsample=0.75,
                                  min_samples_split=900,
                                  min_samples_leaf=50,
                                  max_depth=7,
                                  max_features=10,
                                  random_state=10)
gbm6.fit(X_train,y_train)

y_pred = gbm6.predict(X_train)
y_predprob = gbm6.predict_proba(X_train)
metrics.accuracy_score(y_train,y_pred)
metrics.roc_auc_score(y_train, y_predprob[:, 1])