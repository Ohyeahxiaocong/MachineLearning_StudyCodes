# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:57:33 2019

@author: WZX
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 将训练集分为输入和输出
train = pd.read_csv('dataset/train_modified.csv')
x_colums = [x for x in train.columns if x not in ['id', 'happiness']]
X = train[x_colums]
y = train['happiness'] - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

X = np.array(X_train)
Y = np.array(y_train)

model = keras.Sequential([keras.layers.Dense(128, activation=tf.nn.softmax),
                          keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, Y, epochs=2)
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)
print(metrics.accuracy_score(y_train, y_train_pred))
print(metrics.accuracy_score(y_test, y_test_pred))

#####################################
