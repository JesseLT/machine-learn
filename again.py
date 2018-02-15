#!/usr/bin/python
# coding:utf-8

import numpy as np
from matplotlib import pyplot as plt

def deal_data(x):
        mid = (x.max()+x.min())/2
        mea = (x.max()-x.min())/2
        return np.array([(i-mid)/mea for i in x])

def norm2(x):
        return np.array([i/np.sqrt(np.sum(i*i)) for i in x])

def normw(x):
        return np.array([norm2(i) for i in x])

def get_winners(winxy, scope, r):
        l = list()
        for i in range(scope[0]):
                for j in range(scope[1]):
                        dist = np.sqrt((winxy[0]-i)**2+(winxy[1]-j)**2)
                        if dist<r:
                                l.append([i, j])
        return l

data = np.array([[5, 5],[1.3, 1.5], [6, 6], [5, 6], [6, 5], [1, 1], [2, 2], [1, 2], [2, 1], [-1, 10], [-1, 10.2], [-1.5, 10.5], [-1.8, 11]])
w = np.random.random((2, 2, data.shape[1]))
lrate = 0.9
radius = 3
epochs = 1000

data1 = deal_data(data)
data2 = norm2(data1)
w = normw(w)

for i in range(epochs):
        rindex = np.random.randint(data2.shape[0])
        d = data2[rindex]
        d2 = np.atleast_2d(d)

        o = np.array([j.dot(d2.T) for j in w])#(2, 2, 1)
        win_arr = np.where(o==o.max())
        winxy = win_arr[0][0], win_arr[1][0], win_arr[2][0]
        winners = get_winners(winxy, w.shape, radius)

        for m in winners:
                w[m[0]][m[1]] = w[m[0]][m[1]] + lrate*(d-w[m[0]][m[1]])

        lrate = 0.9*(1-i/(epochs+0.))
        radius = 3*(1-i/(epochs+0.))

colors = ['ro', 'bo', 'yo', 'go']
result = list()
for i in data1:
        r = np.array([j.dot(i.T) for j in w])
        result.append(r)
am = [i.argmax() for i in result]
print(am)
for i in range(data.shape[0]):
        plt.plot(data[i][0], data[i][1], colors[am[i]])
plt.show()
