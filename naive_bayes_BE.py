"""
贝叶斯方法的贝叶斯估计实现
"""

import numpy as np

def naive_bayes_BE_train(x, X, y, xl=None, yl=None, lambdas=0):
    x = np.array(x)
    X = np.array(X)
    y = np.array(y)
    N, n = X.shape
    if not xl:
        xl = [list(set(X[:,i])) for i in range(n)]
    if not yl:
        yl = list(set(y))
    K = len(yl)
    # 统计y各个结果出现的频次
    yf = np.array([(y==ylit).sum() for ylit in yl])
    # 统计x^j,y出现的频次
    xfmatrix = np.zeros((K, n, np.max([len(xlit) for xlit in xl])))
    for i in range(N):
        k = yl.index(y[i])
        for j in range(n):
            h = xl[j].index(X[i][j])
            xfmatrix[k,j,h] += 1
    # 计算出各个类后验概率的分母
    py = (yf + lambdas)/(N+K*lambdas)
    pxmatrix = np.array([[(xfmatrix[k, j, :]+lambdas) / (yf[k] + len(xl[j]) * lambdas) for j in range(n)] for k in range(K)])
    pyx = np.zeros(K)
    for k in range(K):
        pkx = py[k]
        for j in range(n):
            pkx *= pxmatrix[k, j, xl[j].index(x[j])]
        pyx[k] = pkx
    return yl[np.argmax(pyx)]


# 测试

def test_naive_bayes_BE_train():
    X = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'],
         [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'],
         [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']
        ]
    y = [-1, -1, 1, 1, -1, -1, -1,
         1, 1, 1, 1, 1, 1, 1, -1]
    xi = [2, 'S']
    yo = naive_bayes_BE_train(xi, X, y, lambdas=1)
    if yo == -1:
        print('test naive_bayes_BE_train successed!')
    else:
        print('test naive_bayes_BE_train failed!')


if __name__ == '__main__':
    test_naive_bayes_BE_train()
