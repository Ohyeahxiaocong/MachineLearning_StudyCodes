# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:40:51 2019

@author: WZX
"""
import pandas as pd
import numpy as np

train = pd.read_csv('dataset/happiness_train_complete.csv', engine='python')
test = pd.read_csv('dataset/happiness_test_complete.csv', engine='python')

# 找出并删除缺失值或未填写个数大于30个的数据样本
train0 = pd.read_csv('dataset/happiness_train_complete.csv', engine='python')
train0.replace(-8, np.nan, inplace=True)
train0.replace(-1, np.nan, inplace=True)
train0.replace(-2, np.nan, inplace=True)
train0.replace(-3, np.nan, inplace=True)
train0 = 140 - train0.count(axis=1)
train0 = train0[train0 >= 30]

test['type'] = 'test'
train['type'] = 'train'
dataset = pd.concat([test, train], ignore_index=True, sort=False)

# 缺失值统计
nulldata = dataset.apply(lambda x: sum(x.isnull()))

# 数据预处理
dataset.drop(labels='survey_time', axis=1, inplace=True)

dataset['age'] = dataset['birth'].apply(lambda x: 2015 - x)
dataset.drop(labels='birth', axis=1, inplace=True)

dataset['nationality_is_han'] = dataset['nationality'].apply(lambda x: 1 if x == 1 else 0)
dataset.drop(labels='nationality', axis=1, inplace=True)

dataset['religion'].replace(-8, 1, inplace=True)

dataset['religion_freq'].replace(-8, 1, inplace=True)
dataset['religion_freq'] = dataset['religion_freq'].apply(lambda x: 1 if x in [1, 2, 3] else 2 if x in [4, 5, 6] else 3)

dataset['edu'] = dataset['edu'].apply(lambda x: 1 if x ==1 else 2 if x in [2, 3] 
else 3 if x ==4 else 4 if x in [5, 6, 7, 8] else 5 if x in [9, 10] else 6 if x in [11, 12, 13] else 4)

dataset.drop(labels='edu_other', axis=1, inplace=True)

dataset['edu_status'].fillna(-8, inplace=True)
dataset['edu_status'].replace(-8, 5, inplace=True)

dataset['income'] = dataset['income'].apply(lambda x: dataset['income'].median() if x in [-1, -2, -3] else x)

dataset['political'].replace(-8, np.nan, inplace=True)
dataset['political'].fillna(method='ffill', inplace=True)

dataset.drop(labels='join_party', axis=1, inplace=True)





dataset['health'].replace(-8, np.nan, inplace=True)
dataset['health'].fillna(method='ffill', inplace=True)

dataset['health_problem'].replace(-8, np.nan, inplace=True)
dataset['health_problem'].fillna(method='ffill', inplace=True)

dataset['depression'].replace(-8, np.nan, inplace=True)
dataset['depression'].fillna(method='ffill', inplace=True)

dataset['hukou'] = dataset['hukou'].apply(lambda x: 1 if x == 1 else 0)

dataset['hukou_loc'].fillna(method='ffill', inplace=True)
dataset['hukou_loc'] = dataset['hukou_loc'].apply(lambda x: 3 if x in [3, 4] else x)

dataset['media_1'].replace(-8, np.nan, inplace=True)
dataset['media_1'].fillna(method='ffill', inplace=True)
dataset['media_2'].replace(-8, np.nan, inplace=True)
dataset['media_2'].fillna(method='ffill', inplace=True)
dataset['media_3'].replace(-8, np.nan, inplace=True)
dataset['media_3'].fillna(method='ffill', inplace=True)
dataset['media_4'].replace(-8, np.nan, inplace=True)
dataset['media_4'].fillna(method='ffill', inplace=True)
dataset['media_5'].replace(-8, np.nan, inplace=True)
dataset['media_5'].fillna(method='ffill', inplace=True)
dataset['media_6'].replace(-8, np.nan, inplace=True)
dataset['media_6'].fillna(method='ffill', inplace=True)
dataset['media'] = (dataset['media_1'] + dataset['media_2'] + dataset['media_3']
 + dataset['media_4'] + dataset['media_5'] + dataset['media_6'])/6
dataset.drop(labels='media_1', axis=1, inplace=True)
dataset.drop(labels='media_2', axis=1, inplace=True)
dataset.drop(labels='media_3', axis=1, inplace=True)
dataset.drop(labels='media_4', axis=1, inplace=True)
dataset.drop(labels='media_5', axis=1, inplace=True)
dataset.drop(labels='media_6', axis=1, inplace=True)

dataset['leisure_1'].replace(-8, np.nan, inplace=True)
dataset['leisure_1'].fillna(method='ffill', inplace=True)
dataset['leisure_2'].replace(-8, np.nan, inplace=True)
dataset['leisure_2'].fillna(method='ffill', inplace=True)
dataset['leisure_3'].replace(-8, np.nan, inplace=True)
dataset['leisure_3'].fillna(method='ffill', inplace=True)
dataset['leisure_4'].replace(-8, np.nan, inplace=True)
dataset['leisure_4'].fillna(method='ffill', inplace=True)
dataset['leisure_5'].replace(-8, np.nan, inplace=True)
dataset['leisure_5'].fillna(method='ffill', inplace=True)
dataset['leisure_6'].replace(-8, np.nan, inplace=True)
dataset['leisure_6'].fillna(method='ffill', inplace=True)
dataset['leisure_7'].replace(-8, np.nan, inplace=True)
dataset['leisure_7'].fillna(method='ffill', inplace=True)
dataset['leisure_8'].replace(-8, np.nan, inplace=True)
dataset['leisure_8'].fillna(method='ffill', inplace=True)
dataset['leisure_9'].replace(-8, np.nan, inplace=True)
dataset['leisure_9'].fillna(method='ffill', inplace=True)
dataset['leisure_10'].replace(-8, np.nan, inplace=True)
dataset['leisure_10'].fillna(method='ffill', inplace=True)
dataset['leisure_11'].replace(-8, np.nan, inplace=True)
dataset['leisure_11'].fillna(method='ffill', inplace=True)
dataset['leisure_12'].replace(-8, np.nan, inplace=True)
dataset['leisure_12'].fillna(method='ffill', inplace=True)
dataset['leisure'] = (dataset['leisure_1'] + dataset['leisure_2'] + dataset['leisure_3'] 
+ dataset['leisure_4'] + dataset['leisure_5'] + dataset['leisure_6'] + dataset['leisure_7'] 
+ dataset['leisure_8'] + dataset['leisure_9'] + dataset['leisure_10'] + dataset['leisure_11'] 
+ dataset['leisure_12'])/12
dataset.drop(labels='leisure_1', axis=1, inplace=True)
dataset.drop(labels='leisure_2', axis=1, inplace=True)
dataset.drop(labels='leisure_3', axis=1, inplace=True)
dataset.drop(labels='leisure_4', axis=1, inplace=True)
dataset.drop(labels='leisure_5', axis=1, inplace=True)
dataset.drop(labels='leisure_6', axis=1, inplace=True)
dataset.drop(labels='leisure_7', axis=1, inplace=True)
dataset.drop(labels='leisure_8', axis=1, inplace=True)
dataset.drop(labels='leisure_9', axis=1, inplace=True)
dataset.drop(labels='leisure_10', axis=1, inplace=True)
dataset.drop(labels='leisure_11', axis=1, inplace=True)
dataset.drop(labels='leisure_12', axis=1, inplace=True)

dataset['socialize'].replace(-8, np.nan, inplace=True)
dataset['socialize'].fillna(method='ffill', inplace=True)

dataset['relax'].replace(-8, np.nan, inplace=True)
dataset['relax'].fillna(method='ffill', inplace=True)

dataset['learn'].replace(-8, np.nan, inplace=True)
dataset['learn'].fillna(method='ffill', inplace=True)

dataset['social_neighbor'].replace(-8, np.nan, inplace=True)
dataset['social_neighbor'].fillna(method='ffill', inplace=True)
dataset['social_friend'].replace(-8, np.nan, inplace=True)
dataset['social_friend'].fillna(method='ffill', inplace=True)
dataset['social'] = (dataset['social_neighbor'] + dataset['social_friend'])/2
dataset.drop(labels='social_neighbor', axis=1, inplace=True)
dataset.drop(labels='social_friend', axis=1, inplace=True)

dataset['equity'].replace(-8, np.nan, inplace=True)
dataset['equity'].fillna(method='ffill', inplace=True)





