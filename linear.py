#!/usr/bin/python
# coding:utf-8
'''
使用linear nerve(线性神经网络)解决异或问题
'''
import numpy as np
from matplotlib import pyplot as plt

data = [[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]]
data = np.array([i+[i[1]**2, i[1]*i[2], i[2]**2] for i in data])#加入非线性成分(x1**2, x1*x2, x2**2)
eout = np.array([1, 1, -1, -1])
w = (np.random.random(6)-0.5)*2
lr = 0.2
for i in range(1000):
	fout = data.dot(w.T).T # 激活函数为y = x
	if (np.abs(eout-fout)<0.000001).all():#期望输出与实际输出误差小于0.0001时模型收敛
		print('Counts:', i)
		print(fout)
                print(lr)
		break
	else:
		w = w + lr*(eout-fout).dot(data)


x = np.linspace(-1, 2)
plt.figure()
#画出样本和曲线
a = w[5]
b = w[4]*x + w[2]
c = w[3]*x*x + w[1]*x + w[0]

#正样本
x1 = [1, 0]
y1 = [1, 0]
#负样本
x2 = [0, 1]
y2 = [1, 0]

fx1 = (-b + np.sqrt(b*b-4*a*c))/(2*a)
fx2 = (-b - np.sqrt(b*b-4*a*c))/(2*a)
plt.plot(x, fx1, 'r')
plt.plot(x, fx2, 'r')
plt.plot(x1, y1, 'yo')
plt.plot(x2, y2, 'bo')
plt.show()

