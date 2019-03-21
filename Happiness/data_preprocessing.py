# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:40:51 2019

@author: WZX
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

train = pd.read_csv('dataset/happiness_train_complete.csv', engine='python')
test = pd.read_csv('dataset/happiness_test_complete.csv', engine='python')

train.drop(train[train['happiness'] == -8].index, inplace=True)

test['type'] = 'test'
train['type'] = 'train'
dataset = pd.concat([test, train], ignore_index=True, sort=False)

# 数据值统计
nulldata = dataset.isnull().sum()
data_desc = dataset.describe()

# 数据预处理
dataset['survey_type'].replace(2, 0, inplace=True)

dataset.drop(labels='province', axis=1, inplace=True)

#dataset.drop(labels='city', axis=1, inplace=True)

dataset.drop(labels='county', axis=1, inplace=True)

dataset.drop(labels='survey_time', axis=1, inplace=True)

dataset['gender'].replace(2, 0, inplace=True)

dataset['age'] = dataset['birth'].apply(lambda x: 2015 - x)
dataset.drop(labels='birth', axis=1, inplace=True)
plot_data = dataset.groupby('age').count()['id']
#plt.bar(plot_data.index, plot_data)
dataset['age'] = dataset['age'].apply(lambda x: 1 if x >= 18 and x < 33 else 2 if x >= 33 and x < 44
       else 3 if x >= 44 and x < 54 else 4 if x >= 54 and x < 65 else 5)

dataset['nationality_is_han'] = dataset['nationality'].apply(lambda x: 1 if x in [-8, 1] else 0)
dataset.drop(labels='nationality', axis=1, inplace=True)

dataset['religion'].replace(-8, 1, inplace=True)

dataset['religion_freq'].replace(-8, 1, inplace=True)

dataset['edu'].replace(-8, 4, inplace=True)

dataset.drop(labels='edu_other', axis=1, inplace=True)

dataset['edu_status'].fillna(-8, inplace=True)
dataset['edu_status'].replace(-8, 5, inplace=True)

dataset.drop(labels='edu_yr', axis=1, inplace=True)

dataset['income'] = dataset['income'].apply(lambda x: dataset['income'].median() if x in [-1, -2, -3] else x)
dataset['income_class'] = dataset['income'].apply(lambda x: 0 if x == 0 else 1 if x < 10000
       else 2 if x < 20000 else 3 if x <30000 else 4 if x < 50000 else 5)

dataset['political'].replace(-8, 1, inplace=True)

dataset.drop(labels='join_party', axis=1, inplace=True)

dataset['floor_area'] = dataset['floor_area'].apply(lambda x: 1 if x < 60 else 2 if x < 80
       else 3 if x < 100 else 4 if x < 150 else 5)

dataset['property_other'].fillna(0, inplace=True)
dataset['property_other'] = dataset['property_other'].apply(lambda x: x if x == 0 else 1)

#property_1 = dataset['property_1'] + dataset['property_2']
#property_2 = dataset['property_3']
#property_3 = dataset['property_4']
#property_4 = dataset['property_0'] + dataset['property_7'] + dataset['property_8'] + dataset['property_other']
#property_1 = property_1.apply(lambda x: x if x == 0 else 1)
#property_4 = property_4.apply(lambda x: x if x == 0 else 1)
#dataset['property_1'] = property_1
#dataset['property_2'] = property_1
#dataset['property_3'] = property_1
#dataset['property_4'] = property_1
#dataset.drop(labels='property_0', axis=1, inplace=True)
#dataset.drop(labels='property_5', axis=1, inplace=True)
#dataset.drop(labels='property_6', axis=1, inplace=True)
#dataset.drop(labels='property_7', axis=1, inplace=True)
#dataset.drop(labels='property_8', axis=1, inplace=True)
#dataset.drop(labels='property_other', axis=1, inplace=True)

dataset['body_type'] = dataset['weight_jin']/2/pow(dataset['height_cm']/100, 2)
dataset['body_type'] = dataset['body_type'].apply(lambda x: 1 if x < 18.5 else 2
       if x < 24 else 3 if x < 28 else 4)
dataset.drop(labels='height_cm', axis=1, inplace=True)
dataset.drop(labels='weight_jin', axis=1, inplace=True)

dataset['health'].replace(-8, dataset['health'].median(), inplace=True)

dataset['health_problem'].replace(-8, dataset['health_problem'].median(), inplace=True)

dataset['depression'].replace(-8, dataset['depression'].median(), inplace=True)

dataset['hukou_loc'].fillna(1, inplace=True)

dataset['media_1'].replace(-8, dataset['media_1'].median(), inplace=True)
dataset['media_2'].replace(-8, dataset['media_2'].median(), inplace=True)
dataset['media_3'].replace(-8, dataset['media_3'].median(), inplace=True)
dataset['media_4'].replace(-8, dataset['media_4'].median(), inplace=True)
dataset['media_5'].replace(-8, dataset['media_5'].median(), inplace=True)
dataset['media_6'].replace(-8, dataset['media_6'].median(), inplace=True)

dataset['leisure_1'].replace(-8, dataset['leisure_1'].median(), inplace=True)
dataset['leisure_2'].replace(-8, dataset['leisure_2'].median(), inplace=True)
dataset['leisure_3'].replace(-8, dataset['leisure_3'].median(), inplace=True)
dataset['leisure_4'].replace(-8, dataset['leisure_4'].median(), inplace=True)
dataset['leisure_5'].replace(-8, dataset['leisure_5'].median(), inplace=True)
dataset['leisure_6'].replace(-8, dataset['leisure_6'].median(), inplace=True)
dataset['leisure_7'].replace(-8, dataset['leisure_7'].median(), inplace=True)
dataset['leisure_8'].replace(-8, dataset['leisure_8'].median(), inplace=True)
dataset['leisure_9'].replace(-8, dataset['leisure_9'].median(), inplace=True)
dataset['leisure_10'].replace(-8, dataset['leisure_10'].median(), inplace=True)
dataset['leisure_11'].replace(-8, dataset['leisure_11'].median(), inplace=True)
dataset['leisure_12'].replace(-8, dataset['leisure_12'].median(), inplace=True)

dataset['socialize'].replace(-8, dataset['socialize'].median(), inplace=True)

dataset['relax'].replace(-8, dataset['relax'].median(), inplace=True)

dataset['learn'].replace(-8, dataset['learn'].median(), inplace=True)

dataset['social_neighbor'].replace(-8, np.nan, inplace=True)
dataset['social_neighbor'].fillna(dataset['social_neighbor'].median(), inplace=True)

dataset['social_friend'].replace(-8, np.nan, inplace=True)
dataset['social_friend'].fillna(dataset['social_friend'].median(), inplace=True)

dataset['socia_outing'].replace(-8, dataset['socia_outing'].median(), inplace=True)

dataset['equity'].replace(-8, dataset['equity'].median(), inplace=True)

dataset['class'].replace(-8, dataset['class'].median(), inplace=True)

dataset['class_10_before'].replace(-8, dataset['class_10_before'].median(), inplace=True)

dataset['class_10_after'].replace(-8, dataset['class_10_after'].median(), inplace=True)

dataset['class_14'].replace(-8, dataset['class_14'].median(), inplace=True)

dataset.drop(labels='work_status', axis=1, inplace=True)
dataset.drop(labels='work_yr', axis=1, inplace=True)
dataset.drop(labels='work_type', axis=1, inplace=True)
dataset.drop(labels='work_manage', axis=1, inplace=True)

dataset['insur_1'] = dataset['insur_1'].apply(lambda x : x if x ==1 else 0)
dataset['insur_2'] = dataset['insur_2'].apply(lambda x : x if x ==1 else 0)
dataset['insur_3'] = dataset['insur_3'].apply(lambda x : x if x ==1 else 0)
dataset['insur_4'] = dataset['insur_4'].apply(lambda x : x if x ==1 else 0)

dataset['family_income'].fillna(dataset['family_income'].median(), inplace=True)
dataset['family_income'] = dataset['family_income'].apply(lambda x: dataset['family_income'].median() if x in [-1, -2, -3] else x)
dataset['family_income'] = dataset['family_income'].apply(lambda x: 1 if x < 10000
       else 2 if x < 40000 else 3 if x <60000 else 4 if x < 100000 else 5)

dataset['family_m'] = dataset['family_m'].apply(lambda x: dataset['family_m'].median() if x in [-1, -2, -3] else x)

dataset['family_status'].replace(-8, dataset['family_status'].median(), inplace=True)

dataset['house'] = dataset['house'].apply(lambda x: x if x == 1 else 2 if x > 1 else 0)

dataset['car'] = dataset['car'].apply(lambda x: x if x == 1 else 0)

dataset.drop(labels='invest_0', axis=1, inplace=True)

dataset['invest_other'].fillna(0, inplace=True)
dataset['invest_other'] = dataset['invest_other'].apply(lambda x: x if x == 0 else 1)

dataset['son'] = dataset['son'].apply(lambda x: x if x == 1 else 2 if x > 1 else 0)

dataset['daughter'] = dataset['daughter'].apply(lambda x: x if x == 1 else 2 if x > 1 else 0)

dataset['minor_child'] = dataset['minor_child'].apply(lambda x: x if x == 0 else 1)

dataset['marital'] = dataset['marital'].apply(lambda x : 1 if x in [1, 2] else 2 if x in [3, 4] else 3)

dataset.drop(labels='marital_1st', axis=1, inplace=True)

dataset.drop(labels='s_birth', axis=1, inplace=True)

dataset.drop(labels='marital_now', axis=1, inplace=True)

#dataset['s_edu'].fillna(0, inplace=True)
#dataset['s_edu'].replace(-8, np.nan, inplace=True)
#dataset['s_edu'].fillna(method='ffill', inplace=True)
#dataset['s_edu'] = dataset['s_edu'].apply(lambda x: x if x in [0, 1] else 2 if x in [2, 3] 
#else 3 if x ==4 else 4 if x in [5, 6, 7, 8] else 5 if x in [9, 10] else 6 if x in [11, 12, 13] else 4)
#
#dataset['s_political'].fillna(0, inplace=True)
#dataset['s_political'].replace(-8, np.nan, inplace=True)
#dataset['s_political'].fillna(method='ffill', inplace=True)
#
#dataset['s_hukou'] = dataset['s_hukou'].apply(lambda x: 1 if x == 1 else 0)
#
#dataset['s_income'].fillna(0, inplace=True)
#dataset['s_income'] = dataset['s_income'].apply(lambda x: dataset['s_income'].median() if x in [-1, -2, -3] else x)
#
#dataset['s_work_exper'].fillna(0, inplace=True)
#dataset['s_work_exper'] = dataset['s_work_exper'].apply(lambda x: x if x in [0, 1] else 2 if x in [2, 3] else 3 if x in [4, 5] else 4)
#
#dataset.drop(labels='s_work_status', axis=1, inplace=True)
#
#dataset.drop(labels='s_work_type', axis=1, inplace=True)

dataset.drop(labels='s_edu', axis=1, inplace=True)
dataset.drop(labels='s_political', axis=1, inplace=True)
dataset.drop(labels='s_hukou', axis=1, inplace=True)
dataset.drop(labels='s_income', axis=1, inplace=True)
dataset.drop(labels='s_work_exper', axis=1, inplace=True)
dataset.drop(labels='s_work_status', axis=1, inplace=True)
dataset.drop(labels='s_work_type', axis=1, inplace=True)

#dataset['f_birth'] = dataset['f_birth'].apply(lambda x: 1 if x>0 else 0)
dataset.drop(labels='f_birth', axis=1, inplace=True)

dataset['f_edu'].replace(-8, 1, inplace=True)

dataset['f_political'].replace(-8, 1, inplace=True)

dataset['f_work_14'].replace(-8, 2, inplace=True)

#dataset['m_birth'] = dataset['m_birth'].apply(lambda x: 1 if x>0 else 0)
dataset.drop(labels='m_birth', axis=1, inplace=True)

dataset['m_edu'].replace(-8, 1, inplace=True)

dataset['m_political'].replace(-8, 1, inplace=True)

dataset['m_work_14'].replace(-8, 2, inplace=True)

dataset['status_peer'].replace(-8, dataset['status_peer'].median(), inplace=True)

dataset['status_3_before'].replace(-8, dataset['status_3_before'].median(), inplace=True)

dataset['view'].replace(-8, dataset['view'].median(), inplace=True)

dataset['inc_ability'].replace(-8, dataset['inc_ability'].median(), inplace=True)

dataset['inc_exp'] = dataset['inc_exp'].apply(lambda x: dataset['income'].median() if x in [-1, -2, -3] else x)
dataset['inc_exp'] = (dataset['inc_exp'] - dataset['income']).apply(lambda x: 0 if x <= 10000 else 1)
dataset.drop(labels='income', axis=1, inplace=True)

dataset['trust_1'].replace(-8, dataset['trust_1'].median(), inplace=True)
dataset['trust_2'].replace(-8, dataset['trust_2'].median(), inplace=True)
dataset['trust_3'].replace(-8, dataset['trust_3'].median(), inplace=True)
dataset['trust_4'].replace(-8, dataset['trust_4'].median(), inplace=True)
dataset['trust_5'].replace(-8, dataset['trust_5'].median(), inplace=True)
dataset['trust_6'].replace(-8, dataset['trust_6'].median(), inplace=True)
dataset['trust_7'].replace(-8, dataset['trust_7'].median(), inplace=True)
dataset['trust_8'].replace(-8, dataset['trust_8'].median(), inplace=True)
dataset['trust_9'].replace(-8, dataset['trust_9'].median(), inplace=True)
dataset['trust_10'].replace(-8, dataset['trust_10'].median(), inplace=True)
dataset['trust_11'].replace(-8, dataset['trust_11'].median(), inplace=True)
dataset['trust_12'].replace(-8, dataset['trust_12'].median(), inplace=True)
dataset['trust_13'].replace(-8, dataset['trust_13'].median(), inplace=True)

dataset['neighbor_familiarity'].replace(-8, dataset['neighbor_familiarity'].median(), inplace=True)

dataset['public_service_1'].replace([-2, -3], dataset['public_service_1'].median(), inplace=True)
dataset['public_service_2'].replace([-2, -3], dataset['public_service_2'].median(), inplace=True)
dataset['public_service_3'].replace([-2, -3], dataset['public_service_3'].median(), inplace=True)
dataset['public_service_4'].replace([-2, -3], dataset['public_service_4'].median(), inplace=True)
dataset['public_service_5'].replace([-2, -3], dataset['public_service_5'].median(), inplace=True)
dataset['public_service_6'].replace([-2, -3], dataset['public_service_6'].median(), inplace=True)
dataset['public_service_7'].replace([-2, -3], dataset['public_service_7'].median(), inplace=True)
dataset['public_service_8'].replace([-2, -3], dataset['public_service_8'].median(), inplace=True)
dataset['public_service_9'].replace([-2, -3], dataset['public_service_9'].median(), inplace=True)

# 处理后检查是否还有缺失
nulldata = dataset.isnull().sum()

# 连续数据标准化
std = preprocessing.StandardScaler()
std_col = [
           ]
for col in std_col:
    dataset[col] = std.fit_transform(np.array(dataset[col]).reshape(-1, 1))

# one-hot
le = preprocessing.LabelEncoder()
oh_col = ['city', 'religion_freq', 'edu', 'edu_status', 'political', 'floor_area', 
          'health', 'health_problem', 'depression', 'hukou', 'hukou_loc', 'media_1',
          'media_2', 'media_3', 'media_4', 'media_5', 'media_6', 'leisure_1', 'leisure_2',
          'leisure_3', 'leisure_4', 'leisure_5', 'leisure_6', 'leisure_7', 'leisure_8',
          'leisure_9', 'leisure_10', 'leisure_11', 'leisure_12', 'socialize', 'relax', 'learn',
          'social_neighbor', 'social_friend', 'social_outing', 'equity', 'class',
          'class_10_before', 'class_10_after', 'class_14', 'work_exper', 'family_income',
          'family_m', 'family_status', 'house', 'son', 'daughter', 'marital', 'f_edu',
          'f_political', 'f_work_14', 'm_edu', 'm_political', 'm_work_14', 'status_peer',
          'status_3_before', 'view', 'inc_ability', 'trust_1', 'trust_2', 'trust_3',
          'trust_4', 'trust_5', 'trust_6', 'trust_7', 'trust_8', 'trust_9', 'trust_10',
          'trust_11', 'trust_12', 'trust_13', 'neighbor_familiarity', 'age', 'income_class',
          'body_type']
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