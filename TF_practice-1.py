#!/ur/bin/python
#coding: utf-8

'''
视频来源：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
感谢莫烦老师的教程！
'''

import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1*x_data + 0.3

W = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.zeros([1]))

o = W*x_data + b

loss = tf.reduce_mean(tf.square(y_data-o))
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for s in range(2001):
        sess.run(train)
        if s%20==0:
                print(s, sess.run(W), sess.run(b))
