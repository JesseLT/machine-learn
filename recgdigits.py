#!/usr/bin/python
# coding:utf-8

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

class BPnerve():
	def __init__(self, layers):
		self.v = np.random.random((layers[1]+1, layers[0]+1))*2-1
		self.w = np.random.random((layers[2], layers[1]+1))*2-1

	def train(self, data, label, lr=0.11, epochs=20000):
		#添加偏置值
		ones = np.ones((data.shape[0], 1))
		data = np.column_stack((data, ones))

		label = LabelBinarizer().fit_transform(label)

		for i in range(epochs):
			n = np.random.randint(data.shape[0])
			X = np.atleast_2d(data[n])

			y = self.sigmoid(X.dot(self.v.T))
			o = self.sigmoid(y.dot(self.w.T))

			delta_w = (label[n]-o)*self.dsigm(o)
			delta_v = delta_w.dot(self.w)*self.dsigm(y)

			self.v = self.v + lr*delta_v.T.dot(X)
			self.w = self.w + lr*delta_w.T.dot(y)

	def predict(self, data, label):
		#添加偏置值
		ones = np.ones((data.shape[0], 1))
		data = np.column_stack((data, ones))

		y = self.sigmoid(data.dot(self.v.T))
		o = self.sigmoid(y.dot(self.w.T))
		
		results = np.array(map(lambda x:x.argmax(), o))
		print(data.shape)
		print('errors:', results[results!=label])
		print('rights:', label[results!=label])
		print('accurary:', np.mean(label==results))
		

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def dsigm(self, x):
		return x*(1-x)

if __name__=='__main__':
	digits = load_digits()
	datas = digits.data
	targets = digits.target

	#数据归一化
	datas -= datas.min()
	datas /= datas.max()

	train_datas, test_datas, train_label, test_label = train_test_split(datas, targets)
	bp = BPnerve((64, 100, 10))
	bp.train(train_datas, train_label)
	bp.predict(test_datas, test_label)
