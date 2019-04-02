# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:51:10 2019

@author: Hyomin
"""
import tensorflow as tf
import numpy as np

batch_size = 8

# 输入层到隐藏层 隐藏层到输出层的参数 w1,w2
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 提供一个空间，程序运行时再将数据输入
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
+ (1-y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = np.random.RandomState(10)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    step = 5000
    for i in range(step):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
    
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
    
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print('%d:%f' %(i, total_cross_entropy))
        
    print(sess.run(w1))
    print(sess.run(w2))