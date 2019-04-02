# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 09:17:55 2019

@author: WZX
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 训练集和验证集
dataset = pd.read_csv('dataset/train_modified.csv')
test_dataset = pd.read_csv('dataset/test_modified.csv')
x_colums = [x for x in dataset.columns if x not in ['id','happiness']]
X = dataset[x_colums]
y = dataset['happiness']
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
X_resampled_smote = pd.DataFrame(X_resampled_smote,columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_resampled_smote, y_resampled_smote,
                                                    test_size=0.3, random_state=10)
test_data = test_dataset[x_colums]

xgr = xgb.XGBRegressor()
xgr.fit(X, y)
feature_importances = pd.DataFrame()
feature_importances['col'] = X.columns
feature_importances['imp'] = xgr.feature_importances_
feature_importances.sort_values('imp', inplace=True, ascending=False)

plt.bar(x=feature_importances['col'][:30], height=feature_importances['imp'][:30])

X_train = X_train[feature_importances[feature_importances['imp'] > 0.005]['col']]
X_test = X_test[feature_importances[feature_importances['imp'] > 0.005]['col']]
test_data = test_data[feature_importances[feature_importances['imp'] > 0.005]['col']]

xgr1 = xgb.XGBRegressor(max_depth=5,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=1,
                       subsample=0.8,
                       reg_lambda=1,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
param_test1 = {'n_estimators':range(200, 501, 100)}
gsearch1 = GridSearchCV(xgr1, param_grid=param_test1, scoring='neg_mean_squared_error',
                        iid=False, cv=5, n_jobs=-1)
gsearch1.fit(X_train, y_train, eval_set=[(X_test,y_test)], eval_metric='rmse', verbose=False)
gsearch1.best_params_
y_train_pre = gsearch1.predict(X_train)
y_test_pre = gsearch1.predict(X_test)
print(metrics.mean_squared_error(y_train, y_train_pre))
print(metrics.mean_squared_error(y_test, y_test_pre))

xgr2 = xgb.XGBRegressor(max_depth=5,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=1,
                       subsample=0.8,
                       reg_lambda=1,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
param_test2 = {'max_depth':range(3, 10, 2)}
gsearch2 = GridSearchCV(xgr2, param_grid=param_test2, scoring='neg_mean_squared_error',
                        iid=False, cv=5, n_jobs=-1)
gsearch2.fit(X_train, y_train)
gsearch2.best_params_
y_train_pre = gsearch2.predict(X_train)
y_test_pre = gsearch2.predict(X_test)
print(metrics.mean_squared_error(y_train, y_train_pre))
print(metrics.mean_squared_error(y_test, y_test_pre))

xgr3 = xgb.XGBRegressor(max_depth=7,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=1,
                       subsample=0.8,
                       reg_lambda=1,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
param_test3 = {'min_child_weight':range(1, 51, 10)}
gsearch3 = GridSearchCV(xgr3, param_grid=param_test3, scoring='neg_mean_squared_error',
                        iid=False, cv=5, n_jobs=-1)
gsearch3.fit(X_train, y_train)
gsearch3.best_params_
y_train_pre = gsearch3.predict(X_train)
y_test_pre = gsearch3.predict(X_test)
print(metrics.mean_squared_error(y_train, y_train_pre))
print(metrics.mean_squared_error(y_test, y_test_pre))

xgr4 = xgb.XGBRegressor(max_depth=7,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=11,
                       subsample=0.8,
                       reg_alpha = 0,
                       reg_lambda=1,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
param_test4 = {'gamma':range(0, 5, 1)}
gsearch4 = GridSearchCV(xgr4, param_grid=param_test4, scoring='neg_mean_squared_error',
                        iid=False, cv=5, n_jobs=-1)
gsearch4.fit(X_train, y_train)
gsearch4.best_params_
y_train_pre = gsearch4.predict(X_train)
y_test_pre = gsearch4.predict(X_test)
print(metrics.mean_squared_error(y_train, y_train_pre))
print(metrics.mean_squared_error(y_test, y_test_pre))

xgr5 = xgb.XGBRegressor(max_depth=7,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=11,
                       subsample=0.8,
                       reg_lambda=1,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
param_test5 = {'reg_lambda':range(10, 31, 5)}
gsearch5 = GridSearchCV(xgr5, param_grid=param_test5, scoring='neg_mean_squared_error',
                        iid=False, cv=5, n_jobs=-1)
gsearch5.fit(X_train, y_train)
gsearch5.best_params_
y_train_pre = gsearch5.predict(X_train)
y_test_pre = gsearch5.predict(X_test)
print(metrics.mean_squared_error(y_train, y_train_pre))
print(metrics.mean_squared_error(y_test, y_test_pre))

xgr6 = xgb.XGBRegressor(max_depth=7,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=11,
                       subsample=0.8,
                       reg_lambda=15,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
param_test6 = {'subsample': [0.7, 0.75, 0.8, 0.85, 0.9]}
gsearch6 = GridSearchCV(xgr6, param_grid=param_test6, scoring='neg_mean_squared_error',
                        iid=False, cv=5, n_jobs=-1)
gsearch6.fit(X_train, y_train)
gsearch6.best_params_
y_train_pre = gsearch6.predict(X_train)
y_test_pre = gsearch6.predict(X_test)
print(metrics.mean_squared_error(y_train, y_train_pre))
print(metrics.mean_squared_error(y_test, y_test_pre))

#a=[]
#b=[]
#for i in range(160,210,10):
#    xgr = xgb.XGBRegressor(max_depth=5,
#                           learning_rate=0.1,
#                           n_estimators=i,
#                           gamma=0,
#                           min_child_weight=1,
#                           subsample=0.8,
#                           reg_lambda=1,
#                           objective='reg:linear',
#                           booster='gbtree',
#                           n_jobs=-1,
#                           random_state=10)
#    xgr.fit(X_train, y_train)
#    y_train_pre = xgr.predict(X_train)
#    y_test_pre = xgr.predict(X_test)
#    a.append(metrics.mean_squared_error(y_train, y_train_pre))
#    b.append(metrics.mean_squared_error(y_test, y_test_pre))
    
#xgr = xgb.XGBRegressor(max_depth=5,
#                       learning_rate=0.1,
#                       n_estimators=1000,
#                       gamma=10,
#                       min_child_weight=5,
#                       subsample=0.7,
#                       reg_lambda=15,
#                       objective='reg:linear',
#                       booster='gbtree',
#                       n_jobs=-1,
#                       random_state=10)
xgr = xgb.XGBRegressor(max_depth=7,
                       learning_rate=0.2,
                       n_estimators=200,
                       gamma=0,
                       min_child_weight=11,
                       subsample=0.9,
                       reg_lambda=15,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
xgr.fit(X_train, y_train)

test_y = xgr.predict(test_data)
output = pd.DataFrame()
output['id'] = test_dataset['id']
output['happiness'] = test_y.round()
output['happiness'].replace(6.0, 5.0, inplace=True)