# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:33:43 2019

@author: WZX
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import matplotlib.pylab as plt

train = pd.read_csv('train_modified.csv')
x_columns = [x for x in train.columns if x not in ['Disbursed', 'ID']]
X = train[x_columns]
Y = train['Disbursed']

# oob_score:是否采用袋外样本来评估模型的好坏
rf0 = RandomForestClassifier(oob_score=True)
rf0.fit(X, Y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:, 1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(Y, y_predprob))