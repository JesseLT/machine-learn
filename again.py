#!/usr/bin/python
# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, act_func=lambda x:x):
        W = tf.Variable(tf.random_uniform([in_size, out_size]))
        b = tf.Variable(tf.zeros([1, out_size])+0.1)
        outputs = tf.matmul(inputs, W) +b
        return act_func(outputs)

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.04, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, act_func=tf.nn.relu)
o = add_layer(l1, 10, 1)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-o), 1))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)#将画布分成1行1列的图，选中第一个！
ax.scatter(x_data, y_data)#scatter散点图
plt.ion()#使程序不会暂停
plt.show()
for i in range(1200):
        sess.run(train, feed_dict={xs:x_data, ys:y_data})
        if i%25==0:
                print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
                try:
                        ax.lines.remove(lines[0])
                except Exception:
                        pass
                p = sess.run(o, feed_dict={xs:x_data})
                lines = ax.plot(x_data, p, 'r-', lw=5)
                plt.pause(0.1)
sess.close()
'''
add_subplot相比subplot,更具有面向对象性,可以对每个图片的对象进行操作！
'''
