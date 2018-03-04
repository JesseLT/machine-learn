#!/usr/bin/python
# coding:utf-8
'''
多层神经网络   BP(Back Propagation)神经网络
'''
import numpy as np

data = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]])
d = np.array([[1, 1, 0, 0]])
v = np.random.random((5, 3))*2-1
w = np.random.random((1, 5))*2-1
lr = 0.11
counts = 0

def sigmoid(x):
	return 1/(1+np.exp(-x))

def dsigm(x):
	return x*(1-x)

while True:
	y = sigmoid(data.dot(v.T))
	o = sigmoid(y.dot(w.T))
	if np.max(np.abs(d.T-o))<0.01:
		print('Counts:', counts)
		print(o)
		break
	else:
		delta_w = (d.T-o)*dsigm(o)#4, 1
		delta_v = delta_w.dot(w)*dsigm(y)

		w = w + lr*delta_w.T.dot(y)
		v = v + lr*delta_v.T.dot(data)

		counts += 1
