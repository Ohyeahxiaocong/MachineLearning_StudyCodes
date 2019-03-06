# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#scikit-learn v0.20.3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

#读取Excel里的5组数据 将数据分为输入Xi和输出Yi
n=5
for i in range(n):
    sheetname = 'Sheet' + str(i+1)
    locals()['data' + str(i+1)] = \
    pd.read_excel('', sheet_name=sheetname)#指定文件路径
    locals()['X' + str(i + 1)] = locals()['data' + str(i + 1)][['AT', 'V', 'AP', 'RH']]
    locals()['Y' + str(i + 1)] = locals()['data' + str(i + 1)][['PE']]

#简单交叉验证-留出法 default(75%,25%)
for i in range(n):
    locals()['X' + str(i + 1) + '_train'], locals()['X' + str(i + 1) + '_test'],\
    locals()['Y' + str(i + 1) + '_train'], locals()['Y' + str(i + 1) + '_test']\
    = train_test_split(locals()['X' + str(i + 1)], locals()['Y' + str(i + 1)],test_size = 0.2)

#5组数据汇总
X_train_all = pd.concat([X1_train, X2_train, X3_train, X4_train, X5_train])
Y_train_all = pd.concat([Y1_train, Y2_train, Y3_train, Y4_train, Y5_train])
X_test_all = pd.concat([X1_test, X2_test, X3_test, X4_test, X5_test])
Y_test_all = pd.concat([Y1_test, Y2_test, Y3_test, Y4_test, Y5_test])
X_all = pd.concat([X1, X2, X3, X4, X5])
Y_all = pd.concat([Y1, Y2, Y3, Y4, Y5])

#建立回归模型
linreg = LinearRegression()

#返回测试集的均方误差
def get_lr_MSE(X_train, Y_train, X_test, Y_test, linreg):
    linreg.fit(X_train, Y_train)
    Y_prid = linreg.predict(X_test)
    MSE = mean_squared_error(Y_prid, Y_test)
    return MSE

MSE = []
MSE.append({'X1' : get_lr_MSE(X1_train, Y1_train, X1_test, Y1_test, linreg)})
MSE.append({'X2' : get_lr_MSE(X2_train, Y2_train, X2_test, Y2_test, linreg)})
MSE.append({'X3' : get_lr_MSE(X3_train, Y3_train, X3_test, Y3_test, linreg)})
MSE.append({'X4' : get_lr_MSE(X4_train, Y4_train, X4_test, Y4_test, linreg)})
MSE.append({'X5' : get_lr_MSE(X5_train, Y5_train, X5_test, Y5_test, linreg)})
MSE.append({'X_all' : get_lr_MSE(X_train_all, Y_train_all, X_test_all, Y_test_all, linreg)})
#[{'X1': 19.071917470903266},
# {'X2': 20.716748683170326},
# {'X3': 19.673245631287312},
# {'X4': 22.088693481323798},
# {'X5': 21.933980009012334},
# {'X_all': 20.672267090088617}]
#从结果可以看出均方误差的方差较大
#由于留出法受随机划分结果影响较大，且训练集只包含了部分（本例中80%）的样本数据，反映了一个更小的数据集的分布情况

#输出回归模型的偏移b和参数
print(linreg.intercept_)
print(linreg.coef_)

#返回K折交叉验证的均方误差
def get_lr_cv_MSE(X, Y, linreg):
    Y_prid = cross_val_predict(linreg, X, Y, cv=10)
    MSE = mean_squared_error(Y, Y_prid)
    return MSE

MSE_cv = []
MSE_cv.append({'X1' : get_lr_cv_MSE(X1, Y1, linreg)})
MSE_cv.append({'X2' : get_lr_cv_MSE(X2, Y2, linreg)})
MSE_cv.append({'X3' : get_lr_cv_MSE(X3, Y3, linreg)})
MSE_cv.append({'X4' : get_lr_cv_MSE(X4, Y4, linreg)})
MSE_cv.append({'X5' : get_lr_cv_MSE(X5, Y5, linreg)})
MSE_cv.append({'X_all' : get_lr_cv_MSE(X_all, Y_all, linreg)})
#[{'X1': 20.793672509857537},
# {'X2': 20.786948961953403},
# {'X3': 20.786307963466758},
# {'X4': 20.78928409219013},
# {'X5': 20.7955974619431},
# {'X_all': 20.770748348566865}]
#K折交叉验证的均方误差则相对稳定，可以让小的数据集得到更接近真实的均方误差

#可视化
Y_prid = cross_val_predict(linreg, X_all, Y_all, cv=10)
plt.figure()
plt.scatter(Y_all['PE'], Y_prid)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.plot([Y_all.min(), Y_all.max()], [Y_all.min(), Y_all.max()], 'k--', lw = 4)