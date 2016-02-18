import numpy as np

def perceptron_couple_train(X, y, eta=1):
    X = np.array(X)
    y = np.array(y)
    N = y.shape[0]
    a = np.zeros(N)
    b = 0
    goon = (N != 0)
    Gram = np.dot(X, X.T)
    while goon:
        flag = 0
        for i in range(N):
            if y[i]*(np.dot(a*y, Gram[i]) + b) <= 0:
                flag = 1
                while y[i]*(np.dot(a*y, Gram[i]) + b) <= 0:
                    a[i] += eta
                    b += eta*y[i]
                break
        goon = flag
    w = np.dot(a*y, X)
    return a, w, b

def test_perceptron_couple_train():
    X = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    a, w, b = perceptron_couple_train(X, y)
    if ((a == (2, 0, 5)).all() and
        (w == (1, 1)).all() and b == -3):
        print('test perceptron_couple_train successful!')
    else:
        print('test perceptron_couple_train failed!')

if __name__ == '__main__':
    test_perceptron_couple_train()
