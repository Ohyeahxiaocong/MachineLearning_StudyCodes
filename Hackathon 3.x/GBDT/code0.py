# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:47:39 2019

@author: Hyomin
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

# 将训练集分为输入和输出
train = pd.read_csv('datasets/train_modified.csv')
x_colums = [x for x in train.columns if x not in ['Disbursed']]
X = train[x_colums]
y = train['Disbursed']