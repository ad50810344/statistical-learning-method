import numpy as np

def perceptron_origin_train(X, y, eta=1):
    X = np.array(X)
    y = np.array(y)
    N = y.shape[0]
    w = np.zeros(X.shape[1])
    b = 0
    goon = (N != 0)
    while goon:
        flag = 0
        for i in range(N):
            x = X[i]
            if y[i]*(np.dot(w, x) + b) <= 0:
                flag = 1
                while y[i]*(np.dot(w, x) + b) <= 0:
                    w = w + eta * y[i] * X[i]
                    b = b + eta * y[i]
                break
        goon = flag
    return w, b


def test_perceptron_origin_train():
    X = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    w, b = perceptron_origin_train(X, y)
    if (w == (1, 1)).all() and b == -3:
        print('test perceptron_origin_train sucessful!')
    else:
        print('test perceptron_origin_train fail!')

if __name__ == '__main__':
    test_perceptron_origin_train()
