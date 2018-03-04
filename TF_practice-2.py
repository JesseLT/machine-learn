#!/ur/bin/python
#coding: utf-8

'''
视频来源：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
感谢莫烦老师的教程！
'''

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2,4], [2,4]])

p = tf.matmul(matrix1, matrix2)

###########################################################################
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.global_variables_initializer()

##########################################################################
input1 = tf.placeholder(tf.float32, [2, 2])
input2 = tf.placeholder(tf.float32, [2, 3])
output = tf.matmul(input1, input2)
a = np.arange(4).reshape((2,2))
b = np.arange(6).reshape((2,3))

with tf.Session() as sess:
        r = sess.run(p)
        print(r, type(r), sess.run(output, feed_dict={input1:a, input2:b}))
        sess.run(init)
        for _ in range(10):
                print(sess.run(update))
