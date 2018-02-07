#!/usr/bin/python
#coding:utf-8

import numpy as np
from matplotlib import pyplot as plt
import random

#将数据变成[-1, 1]之间的数值
def deal_data(x):
        mid = (x.max()+x.min())/2
        mea = (x.max()-x.min())/2
        return np.array([(i-mid)/mea for i in x])

#将二维数组归一化(二范数)
def norm2(x):
        return np.array([i/np.sqrt(np.sum(i*i)) for i in x])

def norm_w(w):
        return np.array([norm2(i) for i in w])

def get_winners(win, scope, r):
        res = []
        for i in range(scope[0]):
                for j in range(scope[1]):
                        l = np.sqrt((i-win[0])**2+(j-win[1])**2)
                        if l<r:
                                res.append([i, j])
        return res
data = np.array([[5, 5],[1.3, 1.5], [6, 6], [5, 6], [6, 5], [1, 1], [2, 2], [1, 2], [2, 1]])
lrate = 0.9
radius = 3
epchos = 2000

data1 = deal_data(data)
w = np.array(random.sample(data1, 3)).reshape(3, 1, data1.shape[1])
data1 = norm2(data1)
w = norm_w(w)
for i in range(1, epchos+1):
        index = np.random.randint(data1.shape[0])
        d = data1[index]
        o = np.array([np.atleast_2d(d).dot(j.T) for j in w]).reshape(w.shape[0], w.shape[1])#向量点积3X1
        winxy = o.argmax()%o.shape[0], o.argmax()%o.shape[1]
        winners = get_winners(winxy, o.shape, radius)
        #获胜邻域调整权值
        for j in winners:
                w[j[0]][j[1]] = w[j[0]][j[1]] + lrate*(d-w[j[0]][j[1]])

        #改变获胜邻域的半径
        radius = 3*(1-i/(epchos+1.))

        #改变学习率
        lrate = 0.9*(1-i/(epchos+1.))
        #print(o, lrate, radius)
        #if lrate<0.001:
        #        print('counts:', i)
        #        break

res = np.array([data1.dot(k.T) for k in w]).reshape((data1.shape[0], w.shape[0]))
ws = np.argmax(res, axis=1).tolist()
colors = ['y', 'r', 'b', 'g']
for i in range(data.shape[0]):
        plt.plot(data[i][0], data[i][1], colors[ws[i]]+'o')
plt.show()
