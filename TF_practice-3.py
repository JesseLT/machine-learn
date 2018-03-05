#!/ur/bin/python
#coding: utf-8

'''
视频来源：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
感谢莫烦老师的教程！
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, acf=lambda x:x):
        W = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        res = tf.matmul(inputs, W) + b
        #if acf is None:
        #        outputs = res
        #else:
        #        outputs = acf(res)
        #return outputs
        return acf(res)

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.03, x_data.shape)
y_data = np.square(x_data)- 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer1 = add_layer(xs, 1, 10, acf=tf.nn.relu)
o = add_layer(layer1, 10, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-o),reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()#生成一个画布
ax = fig.add_subplot(111)#1行1列的第一个
ax.scatter(x_data, y_data)#散点图
plt.ion()#使程序不会停止
plt.show()

for i in range(1200):
        sess.run(train, feed_dict={xs:x_data, ys:y_data})
        if i%25==0:
                try:
                        ax.lines.remove(lines[0])# 移除前面的线
                except Exception:
                        pass
                print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
                p = sess.run(o, feed_dict={xs:x_data})
                lines = ax.plot(x_data, p, 'r-', lw=5)
                plt.pause(0.1)#暂停0.1秒
sess.close()
