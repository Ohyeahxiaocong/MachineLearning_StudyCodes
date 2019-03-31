# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:11:28 2019

@author: Hyomin
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# config
input_node = 784
output_node = 10
layer1_node = 500
batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
traing_steps = 30000
moving_average_decay = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.avgrage(weights1))
        + avg_class.avgrage(biases1))
        return tf.matmul(layer1, avg_class.avgrage(weights2)) + avg_class.average(biases2)
    
def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')
    
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1, seed=10))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1, seed=10))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
    
    y = inference(x, None, weights1, biases1, weights2, biases2)
    global_step = tf.Variable(0, trainable=False)
    
    