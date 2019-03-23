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

# 训练集和验证集
dataset = pd.read_csv('dataset/train_modified.csv')
test_dataset = pd.read_csv('dataset/test_modified.csv')
x_colums = [x for x in dataset.columns if x not in ['id','happiness']]
X = dataset[x_colums]
y = dataset['happiness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
test_data = test_dataset[x_colums]

a=[]
b=[]
for i in range(50,160,10):
    xgr = xgb.XGBRegressor(max_depth=5,
                           learning_rate=0.1,
                           n_estimators=i,
                           gamma=0,
                           min_child_weight=1,
                           subsample=0.8,
                           reg_lambda=1,
                           objective='reg:linear',
                           booster='gbtree',
                           n_jobs=-1,
                           random_state=10)
    xgr.fit(X_train, y_train)
    y_train_pre = xgr.predict(X_train)
    y_test_pre = xgr.predict(X_test)
    a.append(metrics.mean_squared_error(y_train, y_train_pre))
    b.append(metrics.mean_squared_error(y_test, y_test_pre))
    
xgr = xgb.XGBRegressor(max_depth=i,
                       learning_rate=0.1,
                       n_estimators=70,
                       gamma=0.4,
                       min_child_weight=3,
                       subsample=0.6,
                       reg_lambda=2,
                       objective='reg:linear',
                       booster='gbtree',
                       n_jobs=-1,
                       random_state=10)
xgr.fit(X, y)

test_y = xgr.predict(test_data)
test_y = test_y.round()

output = test_dataset[['id', 'happiness']]
output['happiness'] = test_y.astype('int')