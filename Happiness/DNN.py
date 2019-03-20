# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:57:33 2019

@author: WZX
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np



# 将训练集分为输入和输出
train = pd.read_csv('dataset/train_modified.csv')
test = pd.read_csv('dataset/test_modified.csv')
x_colums = [x for x in train.columns if x not in ['id', 'happiness']]
X_train = train[x_colums]
y_train = train['happiness'] - 1
X_test = test[x_colums]

X = np.array(X_train)
Y = np.array(y_train)

model = keras.Sequential([keras.layers.Dense(128, activation=tf.nn.sigmoid),
                          keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, Y, epochs=10)