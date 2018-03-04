#!/usr/bin/python
# coding:utf-8

import numpy as np
from matplotlib import pyplot as plt

data = np.array([[1, 1,1], [1, 3, 3], [1, 4, 3]])
w = (np.random.random(3)-0.5)*2
eout = np.array([-1, 1, 1])
lr = 0.1
for i in range(100):
	fout = np.sign(data.dot(w.T)).T
	if (fout==eout).all():
		print('Counts:', i)
		break
	else:
		w = w + lr*(eout-fout).dot(data)
		print(w)

#画出样本和曲线
x = np.linspace(0, 5)
y = -(w[1]/w[2])*x-w[0]/w[2]
plt.figure()
plt.plot(x, y, 'r')
plt.plot([3, 4], [3, 3], 'bo')
plt.plot([1], [1], 'yo')
plt.show()
