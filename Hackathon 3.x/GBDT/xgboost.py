# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:22:36 2019

@author: Hyomin
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# 将训练集分为输入和输出
train = pd.read_csv('datasets/train_modified.csv')
test = pd.read_csv('datasets/test_modified.csv')
x_colums = [x for x in train.columns if x not in ['Disbursed']]
X_train = train[x_colums]
y_train = train['Disbursed']
X_test = test[x_colums]
y_test = test['Disbursed']

xgb1 = XGBClassifier(max_depth=5,
                     learning_rate=0.1,
                     n_estimators=1000,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     scale_pos_weight=1,
                     n_jobs=4,
                     seed=10)