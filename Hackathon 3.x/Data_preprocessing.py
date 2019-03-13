# -*- coding: utf-8 -*-
import pandas as pd

train = pd.read_csv('datasets/Train.csv', engine='python', dtype='str')
test = pd.read_csv('datasets/Test.csv', engine='python', dtype='str')

test['type'] = 'test'
train['type'] = 'train'
dataset = pd.concat([test, train], ignore_index=True, sort=False)
dataset.replace(dataset['LoggedIn'][0], '', inplace=True)

# 缺失值数量统计
dataset.apply(lambda x: sum(x.isin([''])))

# 城市有724个类别，类别太多
dataset.drop(labels='City',axis=1, inplace=True)

# 将出生日期改为年龄
dataset['Age'] = dataset['DOB'].apply(lambda x: 115 - int(x[-2:]))
dataset.drop(labels='DOB',axis=1, inplace=True)

# 用户创建日期对结果是否有影响不明
dataset.drop(labels='Lead_Creation_Date', axis=1, inplace=True)

# 用户雇主姓名对结果是否有影响不明
dataset.drop(labels='Employer_Name',axis=1, inplace=True)

# 根据原文需求删除
dataset.drop(labels='LoggedIn',axis=1, inplace=True)

