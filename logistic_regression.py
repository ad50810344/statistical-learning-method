import numpy as np
from scipy.optimize import minimize

def costfunction(w, X, y):
    N = y.shape[0]
    X_ = np.r_['-1', X, np.ones(N).reshape(N, 1)]
    l = 0
    for i in range(N):
        l += np.log(1 + np.e ** (- y[i] * (np.dot(w, X_[i]))))
    return l

def costfunction_der(w, X, y):
    N = y.shape[0]
    X_ = np.r_['-1', X, np.ones(N).reshape(N, 1)]
    der = np.zeros_like(w)
    for i in range(N):
        ywxi = y[i] * np.dot(w, X_[i])
        der_w += (-ywxi) * np.e **(-ywxi) * y[i] * X_[i] / (1 + np.e ** (-ywxi))
    return der_w

def LR_train(X, y):
    # 将该矩阵进行扩展
    m = X.shape[1]
    w0 = np.zeros(m+1)
    cf = lambda w:  costfunction(w, X, y)
    cost_der = lambda w: costfunction_der(w, X, y)
    opf = minimize(cf, w0, method='BFGS', jac=cost_der)
    return opf.x[:-1], opf.x[-1]

def LR_preception(w, b, x):
    f = np.dot(w, x) + b
    if f >= 0:
        return 1
    else:
        return -1


