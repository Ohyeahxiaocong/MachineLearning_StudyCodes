# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:40:51 2019

@author: WZX
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

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
train0_index = list(train0.index)
train.drop(train0_index, axis=0, inplace=True)

train['happiness'].replace(-8, np.nan, inplace=True)
train['happiness'].fillna(method='ffill', inplace=True)

test['type'] = 'test'
train['type'] = 'train'
dataset = pd.concat([test, train], ignore_index=True, sort=False)

# 缺失值统计
nulldata = dataset.apply(lambda x: sum(x.isnull()))

# 数据预处理

dataset['survey_type'].replace(2, 0, inplace=True)

dataset.drop(labels='province', axis=1, inplace=True)

dataset.drop(labels='city', axis=1, inplace=True)

dataset.drop(labels='county', axis=1, inplace=True)

dataset.drop(labels='survey_time', axis=1, inplace=True)

dataset['gender'].replace(2, 0, inplace=True)

dataset['age'] = dataset['birth'].apply(lambda x: 2015 - x)
dataset.drop(labels='birth', axis=1, inplace=True)

dataset['nationality_is_han'] = dataset['nationality'].apply(lambda x: 1 if x == 1 else 0)
dataset.drop(labels='nationality', axis=1, inplace=True)

dataset['religion'].replace(-8, 1, inplace=True)

dataset['religion_freq'].replace(-8, 1, inplace=True)
#dataset['religion_freq'] = dataset['religion_freq'].apply(lambda x: 1 if x in [1, 2, 3] else 2 if x in [4, 5, 6] else 3)

dataset['edu'] = dataset['edu'].apply(lambda x: 1 if x ==1 else 2 if x in [2, 3] 
else 3 if x ==4 else 4 if x in [5, 6, 7, 8] else 5 if x in [9, 10] else 6 if x in [11, 12, 13] else 4)

dataset.drop(labels='edu_other', axis=1, inplace=True)

dataset['edu_status'].fillna(-8, inplace=True)
dataset['edu_status'].replace(-8, 5, inplace=True)

dataset.drop(labels='edu_yr', axis=1, inplace=True)

dataset['income'] = dataset['income'].apply(lambda x: dataset['income'].median() if x in [-1, -2, -3] else x)

dataset['political'].replace(-8, np.nan, inplace=True)
dataset['political'].fillna(method='ffill', inplace=True)

dataset.drop(labels='join_party', axis=1, inplace=True)

dataset['property_other'].fillna(0, inplace=True)
dataset['property_other'] = dataset['property_other'].apply(lambda x: x if x == 0 else 1)
property_1 = dataset['property_1'] + dataset['property_2']
property_2 = dataset['property_3']
property_3 = dataset['property_4']
property_4 = dataset['property_0'] + dataset['property_7'] + dataset['property_8'] + dataset['property_other']
property_1 = property_1.apply(lambda x: x if x == 0 else 1)
property_4 = property_4.apply(lambda x: x if x == 0 else 1)
dataset['property_1'] = property_1
dataset['property_2'] = property_1
dataset['property_3'] = property_1
dataset['property_4'] = property_1
dataset.drop(labels='property_0', axis=1, inplace=True)
dataset.drop(labels='property_5', axis=1, inplace=True)
dataset.drop(labels='property_6', axis=1, inplace=True)
dataset.drop(labels='property_7', axis=1, inplace=True)
dataset.drop(labels='property_8', axis=1, inplace=True)
dataset.drop(labels='property_other', axis=1, inplace=True)

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

dataset['socia_outing'].replace(-8, np.nan, inplace=True)
dataset['socia_outing'].fillna(method='ffill', inplace=True)

dataset['equity'].replace(-8, np.nan, inplace=True)
dataset['equity'].fillna(method='ffill', inplace=True)

dataset['class'].replace(-8, np.nan, inplace=True)
dataset['class'].fillna(method='ffill', inplace=True)

dataset['class_10_before'].replace(-8, np.nan, inplace=True)
dataset['class_10_before'].fillna(method='ffill', inplace=True)

dataset['class_10_after'].replace(-8, np.nan, inplace=True)
dataset['class_10_after'].fillna(method='ffill', inplace=True)

dataset['class_14'].replace(-8, np.nan, inplace=True)
dataset['class_14'].fillna(method='ffill', inplace=True)

dataset['work_exper'] = dataset['work_exper'].apply(lambda x: x if x == 1 else 2 if x in [2, 3] else 3 if x in [4, 5] else 4)

dataset.drop(labels='work_status', axis=1, inplace=True)
dataset.drop(labels='work_yr', axis=1, inplace=True)
dataset.drop(labels='work_type', axis=1, inplace=True)
dataset.drop(labels='work_manage', axis=1, inplace=True)

dataset['insur_1'].replace(-8, np.nan, inplace=True)
dataset['insur_1'].replace(-1, np.nan, inplace=True)
dataset['insur_1'].fillna(method='ffill', inplace=True)
dataset['insur_1'].replace(2, 0, inplace=True)
dataset['insur_2'].replace(-8, np.nan, inplace=True)
dataset['insur_2'].replace(-1, np.nan, inplace=True)
dataset['insur_2'].fillna(method='ffill', inplace=True)
dataset['insur_2'].replace(2, 0, inplace=True)
dataset['insur_3'].replace(-8, np.nan, inplace=True)
dataset['insur_3'].replace(-1, np.nan, inplace=True)
dataset['insur_3'].fillna(method='ffill', inplace=True)
dataset['insur_3'].replace(2, 0, inplace=True)
dataset['insur_4'].replace(-8, np.nan, inplace=True)
dataset['insur_4'].replace(-1, np.nan, inplace=True)
dataset['insur_4'].fillna(method='ffill', inplace=True)
dataset['insur_4'].replace(2, 0, inplace=True)
dataset['insur'] = dataset['insur_1'] + dataset['insur_2'] + dataset['insur_3'] + dataset['insur_4']
dataset.drop(labels='insur_1', axis=1, inplace=True)
dataset.drop(labels='insur_2', axis=1, inplace=True)
dataset.drop(labels='insur_3', axis=1, inplace=True)
dataset.drop(labels='insur_4', axis=1, inplace=True)

dataset['family_income'].replace(-1, np.nan, inplace=True)
dataset['family_income'].replace(-2, np.nan, inplace=True)
dataset['family_income'].replace(-3, np.nan, inplace=True)
dataset['family_income'].fillna(dataset['family_income'].median(), inplace=True)

dataset['family_m'] = dataset['family_m'].apply(lambda x : x if x in [1, 2, 3, 4, 5] else 6)

dataset['family_status'].replace(-8, np.nan, inplace=True)
dataset['family_status'].fillna(method='ffill', inplace=True)

dataset['house'].replace(-8, np.nan, inplace=True)
dataset['house'].fillna(method='ffill', inplace=True)

dataset['car'].replace(-8, np.nan, inplace=True)
dataset['car'].fillna(method='ffill', inplace=True)

dataset['invest'] = dataset['invest_1']
dataset.drop(labels='invest_0', axis=1, inplace=True)
dataset.drop(labels='invest_1', axis=1, inplace=True)
dataset.drop(labels='invest_2', axis=1, inplace=True)
dataset.drop(labels='invest_3', axis=1, inplace=True)
dataset.drop(labels='invest_4', axis=1, inplace=True)
dataset.drop(labels='invest_5', axis=1, inplace=True)
dataset.drop(labels='invest_6', axis=1, inplace=True)
dataset.drop(labels='invest_7', axis=1, inplace=True)
dataset.drop(labels='invest_8', axis=1, inplace=True)
dataset.drop(labels='invest_other', axis=1, inplace=True)

dataset['son'].replace(-8, np.nan, inplace=True)
dataset['son'].fillna(method='ffill', inplace=True)
dataset['son'] = dataset['son'].apply(lambda x : x if x in [0, 1, 2] else 3)

dataset['daughter'].replace(-8, np.nan, inplace=True)
dataset['daughter'].fillna(method='ffill', inplace=True)
dataset['daughter'] = dataset['daughter'].apply(lambda x : x if x in [0, 1, 2] else 3)

dataset['minor_child'].replace(-8, np.nan, inplace=True)
dataset['minor_child'].fillna(0, inplace=True)
dataset['minor_child'] = dataset['minor_child'].apply(lambda x : x if x == 0 else 1)

dataset['marital'] = dataset['marital'].apply(lambda x : 1 if x in [1, 2] else 2 if x in [3, 4] else 3)

dataset.drop(labels='marital_1st', axis=1, inplace=True)

dataset.drop(labels='s_birth', axis=1, inplace=True)

dataset.drop(labels='marital_now', axis=1, inplace=True)

dataset['s_edu'].fillna(0, inplace=True)
dataset['s_edu'].replace(-8, np.nan, inplace=True)
dataset['s_edu'].fillna(method='ffill', inplace=True)
dataset['s_edu'] = dataset['s_edu'].apply(lambda x: x if x in [0, 1] else 2 if x in [2, 3] 
else 3 if x ==4 else 4 if x in [5, 6, 7, 8] else 5 if x in [9, 10] else 6 if x in [11, 12, 13] else 4)

dataset['s_political'].fillna(0, inplace=True)
dataset['s_political'].replace(-8, np.nan, inplace=True)
dataset['s_political'].fillna(method='ffill', inplace=True)

dataset['s_hukou'] = dataset['s_hukou'].apply(lambda x: 1 if x == 1 else 0)

dataset['s_income'].fillna(0, inplace=True)
dataset['s_income'] = dataset['s_income'].apply(lambda x: dataset['s_income'].median() if x in [-1, -2, -3] else x)

dataset['s_work_exper'].fillna(0, inplace=True)
dataset['s_work_exper'] = dataset['s_work_exper'].apply(lambda x: x if x in [0, 1] else 2 if x in [2, 3] else 3 if x in [4, 5] else 4)

dataset.drop(labels='s_work_status', axis=1, inplace=True)

dataset.drop(labels='s_work_type', axis=1, inplace=True)

dataset['f_birth'] = dataset['f_birth'].apply(lambda x: 1 if x>0 else 0)

dataset['f_edu'].replace(-8, np.nan, inplace=True)
dataset['f_edu'].fillna(method='ffill', inplace=True)
dataset['f_edu'] = dataset['f_edu'].apply(lambda x: x if x in [0, 1] else 2 if x in [2, 3] 
else 3 if x ==4 else 4 if x in [5, 6, 7, 8] else 5 if x in [9, 10] else 6 if x in [11, 12, 13] else 4)

dataset['f_political'].replace(-8, np.nan, inplace=True)
dataset['f_political'].fillna(method='ffill', inplace=True)

dataset['f_work_14'].replace(-8, np.nan, inplace=True)
dataset['f_work_14'] = dataset['f_work_14'].apply(lambda x: 0 if x in [11, 12, 13, 14, 15, 16, 17] else 1)

dataset['m_birth'] = dataset['m_birth'].apply(lambda x: 1 if x>0 else 0)

dataset['m_edu'].replace(-8, np.nan, inplace=True)
dataset['m_edu'].fillna(method='ffill', inplace=True)
dataset['m_edu'] = dataset['m_edu'].apply(lambda x: x if x in [0, 1] else 2 if x in [2, 3] 
else 3 if x ==4 else 4 if x in [5, 6, 7, 8] else 5 if x in [9, 10] else 6 if x in [11, 12, 13] else 4)

dataset['m_political'].replace(-8, np.nan, inplace=True)
dataset['m_political'].fillna(method='ffill', inplace=True)

dataset['m_work_14'].replace(-8, np.nan, inplace=True)
dataset['m_work_14'] = dataset['m_work_14'].apply(lambda x: 0 if x in [11, 12, 13, 14, 15, 16, 17] else 1)

dataset['status_peer'].replace(-8, np.nan, inplace=True)
dataset['status_peer'].fillna(method='ffill', inplace=True)

dataset['status_3_before'].replace(-8, np.nan, inplace=True)
dataset['status_3_before'].fillna(method='ffill', inplace=True)

dataset['view'].replace(-8, np.nan, inplace=True)
dataset['view'].fillna(method='ffill', inplace=True)

dataset['inc_ability'].replace(-8, np.nan, inplace=True)
dataset['inc_ability'].fillna(method='ffill', inplace=True)

dataset['inc_exp'] = dataset['inc_exp'].apply(lambda x: dataset['income'].median() if x in [-1, -2, -3] else x)
dataset['inc_exp'] = (dataset['inc_exp'] - dataset['income']).apply(lambda x: 0 if x <= 10000 else 1)

dataset['trust_1'].replace(-8, np.nan, inplace=True)
dataset['trust_1'].fillna(method='ffill', inplace=True)
dataset['trust_2'].replace(-8, np.nan, inplace=True)
dataset['trust_2'].fillna(method='ffill', inplace=True)
dataset['trust_3'].replace(-8, np.nan, inplace=True)
dataset['trust_3'].fillna(method='ffill', inplace=True)
dataset['trust_4'].replace(-8, np.nan, inplace=True)
dataset['trust_4'].fillna(method='ffill', inplace=True)
dataset['trust_5'].replace(-8, np.nan, inplace=True)
dataset['trust_5'].fillna(method='ffill', inplace=True)
dataset['trust_6'].replace(-8, np.nan, inplace=True)
dataset['trust_6'].fillna(method='ffill', inplace=True)
dataset['trust_7'].replace(-8, np.nan, inplace=True)
dataset['trust_7'].fillna(method='ffill', inplace=True)
dataset['trust_8'].replace(-8, np.nan, inplace=True)
dataset['trust_8'].fillna(method='ffill', inplace=True)
dataset['trust_9'].replace(-8, np.nan, inplace=True)
dataset['trust_9'].fillna(method='ffill', inplace=True)
dataset['trust_10'].replace(-8, np.nan, inplace=True)
dataset['trust_10'].fillna(method='ffill', inplace=True)
dataset['trust_11'].replace(-8, np.nan, inplace=True)
dataset['trust_11'].fillna(method='ffill', inplace=True)
dataset['trust_12'].replace(-8, np.nan, inplace=True)
dataset['trust_12'].fillna(method='ffill', inplace=True)
dataset['trust_13'].replace(-8, np.nan, inplace=True)
dataset['trust_13'].fillna(method='ffill', inplace=True)
dataset['trust'] = (dataset['trust_1'] + dataset['trust_2'] + dataset['trust_3'] + dataset['trust_4']
+ dataset['trust_5'] + dataset['trust_6'] + dataset['trust_7'] + dataset['trust_8'] + dataset['trust_9']
+ dataset['trust_10'] + dataset['trust_11'] + dataset['trust_12'] + dataset['trust_13'])/13
dataset.drop(labels='trust_1', axis=1, inplace=True)
dataset.drop(labels='trust_2', axis=1, inplace=True)
dataset.drop(labels='trust_3', axis=1, inplace=True)
dataset.drop(labels='trust_4', axis=1, inplace=True)
dataset.drop(labels='trust_5', axis=1, inplace=True)
dataset.drop(labels='trust_6', axis=1, inplace=True)
dataset.drop(labels='trust_7', axis=1, inplace=True)
dataset.drop(labels='trust_8', axis=1, inplace=True)
dataset.drop(labels='trust_9', axis=1, inplace=True)
dataset.drop(labels='trust_10', axis=1, inplace=True)
dataset.drop(labels='trust_11', axis=1, inplace=True)
dataset.drop(labels='trust_12', axis=1, inplace=True)
dataset.drop(labels='trust_13', axis=1, inplace=True)
dataset['trust'].fillna(dataset['trust'].mean(), inplace=True)

dataset['neighbor_familiarity'].replace(-8, np.nan, inplace=True)
dataset['neighbor_familiarity'].fillna(method='ffill', inplace=True)

dataset['public_service_1'].replace(-2, np.nan, inplace=True)
dataset['public_service_1'].replace(-3, np.nan, inplace=True)
dataset['public_service_1'].fillna(method='ffill', inplace=True)
dataset['public_service_2'].replace(-2, np.nan, inplace=True)
dataset['public_service_2'].replace(-3, np.nan, inplace=True)
dataset['public_service_2'].fillna(method='ffill', inplace=True)
dataset['public_service_3'].replace(-2, np.nan, inplace=True)
dataset['public_service_3'].replace(-3, np.nan, inplace=True)
dataset['public_service_3'].fillna(method='ffill', inplace=True)
dataset['public_service_4'].replace(-2, np.nan, inplace=True)
dataset['public_service_4'].replace(-3, np.nan, inplace=True)
dataset['public_service_4'].fillna(method='ffill', inplace=True)
dataset['public_service_5'].replace(-2, np.nan, inplace=True)
dataset['public_service_5'].replace(-3, np.nan, inplace=True)
dataset['public_service_5'].fillna(method='ffill', inplace=True)
dataset['public_service_6'].replace(-2, np.nan, inplace=True)
dataset['public_service_6'].replace(-3, np.nan, inplace=True)
dataset['public_service_6'].fillna(method='ffill', inplace=True)
dataset['public_service_7'].replace(-2, np.nan, inplace=True)
dataset['public_service_7'].replace(-3, np.nan, inplace=True)
dataset['public_service_7'].fillna(method='ffill', inplace=True)
dataset['public_service_8'].replace(-2, np.nan, inplace=True)
dataset['public_service_8'].replace(-3, np.nan, inplace=True)
dataset['public_service_8'].fillna(method='ffill', inplace=True)
dataset['public_service_9'].replace(-2, np.nan, inplace=True)
dataset['public_service_9'].replace(-3, np.nan, inplace=True)
dataset['public_service_9'].fillna(method='ffill', inplace=True)
dataset['public_service'] = (dataset['public_service_1'] + dataset['public_service_2'] + dataset['public_service_3']
+ dataset['public_service_4'] + dataset['public_service_5'] + dataset['public_service_6']
+ dataset['public_service_7'] + dataset['public_service_8'] + dataset['public_service_9'])/9
dataset.drop(labels='public_service_1', axis=1, inplace=True)
dataset.drop(labels='public_service_2', axis=1, inplace=True)
dataset.drop(labels='public_service_3', axis=1, inplace=True)
dataset.drop(labels='public_service_4', axis=1, inplace=True)
dataset.drop(labels='public_service_5', axis=1, inplace=True)
dataset.drop(labels='public_service_6', axis=1, inplace=True)
dataset.drop(labels='public_service_7', axis=1, inplace=True)
dataset.drop(labels='public_service_8', axis=1, inplace=True)
dataset.drop(labels='public_service_9', axis=1, inplace=True)

# 处理后检查是否还有缺失
nulldata = dataset.apply(lambda x: sum(x.isnull()))

# 连续数据标准化
std = preprocessing.StandardScaler()
std_col = ['income', 'floor_area', 'height_cm', 'weight_jin', 'family_income',
           'age', 'media', 'leisure', 'social', 'trust', 'public_service',
           'socialize', 'relax', 'learn', 'religion_freq', 'health', 'health_problem',
           'depression', 'socia_outing', 'equity', 'class', 'class_10_before',
           'class_10_after', 'class_14', 'family_status', 'house', 'car', 's_income',
           's_work_exper', 'status_peer', 'status_3_before', 'view', 'inc_ability',
           'neighbor_familiarity', 'insur']
for col in std_col:
    dataset[col] = std.fit_transform(np.array(dataset[col]).reshape(-1, 1))

# one-hot
le = preprocessing.LabelEncoder()
oh_col = ['edu', 'edu_status', 'political', 'hukou_loc', 'work_exper', 'family_m',
          'son', 'daughter', 'marital', 's_edu', 's_political', 'f_edu',
          'f_political', 'm_edu', 'm_political']
for col in oh_col:
    dataset[col] = le.fit_transform(dataset[col])
dataset = pd.get_dummies(dataset, columns=oh_col)

# 导出数据集
train = dataset[dataset['type'] == 'train']
train.drop(labels='type', axis=1, inplace=True)
train.to_csv('dataset/train_modified.csv', encoding='utf-8', index=0)

test = dataset[dataset['type'] == 'test']
test.drop(labels='type', axis=1, inplace=True)
test.to_csv('dataset/test_modified.csv', encoding='utf-8', index=0)