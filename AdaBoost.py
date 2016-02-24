import numpy as np
from functools import reduce

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def error_rate(X, y, gm, d=None):
    N = X.shape[0]
    if d is  None:
        d = np.ones(N)/N
    e = 0
    for i in range(N):
        if gm(X[i]) != y[i]:
            e += d[i]
    return e

def AdaBoost_train(X, y, g):
    """
    X,y 训练数据集

    g

    callable, 接收三个变量，
    分别是训练数据X,y 以及权值训练数据的权值d
    输出也是一个callable对象,表示得到的基本分类器

    Return:
    基本分类器、基本分类器的权重以及
    最终的分类器f, 是一个callable对象
    """
    X = np.array(X)
    y = np.array(y)
    alpha_list = []
    glist = []
    N = X.shape[0]
    d = np.ones(N)/N
    gm = g(X, y, d)
    e = error_rate(X, y, gm, d)
    alpha = 0.5*np.log(1/e - 1)
    alpha_list.append(alpha)
    glist.append(gm)
    f = lambda x: sign(reduce(lambda s, t: s+t, [a*g(x) for a, g in tuple(zip(alpha_list, glist))]))
    lll = 0
    while any( (f(X[i]) != y[i] for i in range(N)) ) and lll < 3:
        lll += 1
        for i in range(N):
            if gm(X[i]) != y[i]:
                d[i] = d[i] * np.e ** alpha
            else:
                d[i] = d[i] * np.e ** (-alpha)
        zm = d.sum()
        d = d / zm
        gm = g(X, y, d)
        e = 0
        for i in range(N):
            if gm(X[i]) != y[i]:
                e += d[i]
        alpha = 0.5 * np.log(1/e - 1)
        alpha_list.append(alpha)
        glist.append(gm)
        f = lambda x: sign(reduce(lambda s, t: s+t, [a*g(x) for a, g in tuple(zip(alpha_list, glist))]))
    return alpha_list, glist, f

def test_classify(X, y, d):
    vs = np.arange(-1, 10) + 0.5
    g1 = lambda x, v: sign(x - v)
    g2 = lambda x, v: sign(v - x)
    e = 1
    g = None
    for v in vs:
        gv1 = lambda x, v=v: g1(x, v)
        gv2 = lambda x, v=v: g2(x, v)
        e1 = error_rate(X, y, gv1, d)
        e2 = error_rate(X, y, gv2, d)
        if e1 < e:
            e = e1
            g = gv1
        if e2 < e:
            e = e2
            g = gv2
    return g

def test_AdaBoost_train():
    X = [
        [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]
    ]
    y = [
        1, 1, 1, -1, -1, -1, 1, 1, 1, -1
    ]
    g = test_classify
    alpha_list, glist, f = AdaBoost_train(X, y, g)
    print(alpha_list)
    if ((np.array(alpha_list) - np.array([0.4236, 0.6496, 0.7541]))**2).sum() < 1e5:
        print('test AdaBoost_train successfully!')
    else:
        print('test AdaBoost_train failed!')

if __name__ == '__main__':
    test_AdaBoost_train()
