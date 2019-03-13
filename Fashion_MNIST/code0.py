# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
# 获得四个numpy.ndarray
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])#x轴线
    plt.yticks([])#y轴线
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])#x轴标签

# keras.Sequential Linear stack of layers 线性堆叠层
# keras.layers.Flatten 将28*28的矩阵变为28*28长的一维向量 activation默认为 f(x)=x
# keras.layers.Dense 常规的密链接NN层
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)])

# keras.Sequential.compile 配置训练模型
# tf.train.AdamOptimizer() Adam方法优化参数
# loss 损失函数 'categorical_crossentropy' 交叉熵损失函数 分类需要one-hot编码 
# 'sparse_categorical_crossentropy' 不需要对分类编码
# metrics 评估方法 通常用精确率metrics=['accuracy']
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# keras.Sequential.fit 依据固定次数用训练集训练模型 epochs为所有数据迭代(前项+反向)次数
model.fit(train_images, train_labels, epochs=5)

# keras.Sequential.evaluate 返回测试集的损失值和准确值
test_loss, test_acc = model.evaluate(test_images, test_labels)

# keras.Sequential.predict 对测试集进行预测，输出预测结果
predictions = model.predict(test_images)
# more about keras.Sequential:
# https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/keras/models/Sequential#compile

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
             100*np.max(predictions_array),class_names[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')