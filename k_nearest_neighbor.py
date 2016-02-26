import numpy as np

def kNN_train(X, y, x, k):
    X = np.array(X)
    y = np.array(y)
    x = np.array(x)
    yk = y[np.argsort(((X-x)**2).sum(axis=1)) < k]
    yunique = np.unique(yk)
    yend = yunique[(yk - yunique[:, np.newaxis]).sum(axis=1).argmax()]
    return yend

def test_kNN_train():
    X = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 5],
        [8, 9],
        [4, 3]
    ]
    y = [
        1, 1, 1, -1, -1, -1
    ]
    x = [4.5, 4.5]
    ypredit = kNN_train(X, y, x, 3)
    if ypredit == -1:
        print('test kNN_train sucessfully')
    else:
        print('test kNN_train failed!')

if __name__ == '__main__':
    test_kNN_train()
