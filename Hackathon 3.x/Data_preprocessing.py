# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('datasets/Train.csv', engine='python')
#test = pd.read_csv('datasets/Test.csv', engine='python')

#test['type'] = 'test'
#train['type'] = 'train'
#dataset = pd.concat([test, train], ignore_index=True, sort=False)

dataset = train

# 缺失值数量统计
nulldata = dataset.apply(lambda x: sum(x.isnull()))

# 城市有724个类别，类别太多,切不清楚这写城市对结果的影响，因此删除该特征
len(dataset['City'].drop_duplicates())
dataset.drop(labels='City', axis=1, inplace=True)

# 将出生日期改为年龄
dataset['Age'] = dataset['DOB'].apply(lambda x: 115 - int(x[-2:]))
dataset.drop(labels='DOB',axis=1, inplace=True)

# 贷款申请日期 删除
dataset.drop(labels='Lead_Creation_Date', axis=1, inplace=True)

# 贷款申请金额 111个缺失值 根据箱线图得知异常值数据分布倾斜 用中位数填充
dataset.boxplot(column='Loan_Amount_Applied')
dataset['Loan_Amount_Applied'].fillna(value=dataset['Loan_Amount_Applied'].median(), inplace=True)

# 贷款年限缺失值处理同贷款金额
dataset['Loan_Tenure_Applied'].fillna(value=dataset['Loan_Tenure_Applied'].median(), inplace=True)

# 当前贷款缺失值处理同金额
dataset['Existing_EMI'].fillna(value=dataset['Existing_EMI'].median(), inplace=True)

# 用户雇主5W多类 删除
dataset.drop(labels='Employer_Name', axis=1, inplace=True)

# 工资账户银行 根据数据分布和缺失值的数量 分为6类 数量排名前5的各为一类 缺失的为一类 剩余的为一类 一共7类
# 输入数据集以及分类个数n
def Salary_Account_Process_Fuc(dataset, n):
    type0 = dataset.groupby('Salary_Account').count()['ID'].sort_values(ascending=False)[: n-2]
    type_list = []
    for i in dataset['Salary_Account']:
        if pd.isnull(i):
            type_list.append(n)
        elif i in type0.index:
            for j in range(n-2):
                if i == type0.index[j]:
                    type_list.append(j)
        else:
            type_list.append(n-1)
    return type_list

dataset['Salary_Account_Type'] = Salary_Account_Process_Fuc(dataset, 7)
dataset.drop(labels='Salary_Account', axis=1, inplace=True)

# 获得贷款资格后申请的贷款金额 缺失值认定为没有申请贷款 可将用户分为贷款和没贷款两类
dataset['Loan_or_Not'] = dataset['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)

# 实际贷款金额 未贷款的缺失值设为0
dataset['Loan_Amount_Submitted'].fillna(value=0, inplace=True)

# 获得贷款资格后申请的贷款年限处理如上
dataset['Loan_Tenure_Submitted'].fillna(value=0, inplace=True)

# 利率 手续费 EMI贷款提交的缺失值占了70%左右 将其分为有和没有两类
dataset['Interest_Rate_Missing'] = dataset['Interest_Rate'].apply(lambda x: 1 if pd.isnull(x) else 0)
dataset.drop(labels='Interest_Rate', axis=1, inplace=True)
dataset['Processing_Fee_Missing'] = dataset['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
dataset.drop(labels='Processing_Fee', axis=1, inplace=True)
dataset['EMI_Loan_Submitted_Missing'] = dataset['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
dataset.drop(labels='EMI_Loan_Submitted', axis=1, inplace=True)

# 根据原文需求删除
dataset.drop(labels='LoggedIn', axis=1, inplace=True)

# One-hot编码 
le = LabelEncoder()
var_to_encode = ['Gender', 'Mobile_Verified', 'Var1', 'Filled_Form', 'Device_Type', 'Var2', 'Source', 'Var4']
for col in var_to_encode:
    dataset[col] = le.fit_transform(dataset[col])
dataset = pd.get_dummies(dataset, columns=var_to_encode)

# 输出处理后的训练集和测试集
train = dataset[dataset['type'] == 'train']
train.drop(labels='type', axis=1, inplace=True)
train.to_csv('datasets/train_modified.csv', encoding='utf-8', index=0)

test = dataset[dataset['type'] == 'test']
test.drop(labels='type', axis=1, inplace=True)
test.to_csv('datasets/test_modified.csv', encoding='utf-8', index=0)