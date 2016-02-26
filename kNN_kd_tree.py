import numpy as np

class kdTree(object):
    def __init__(self, left=None, right=None, data=np.array([]), depth=0, seg=None, father=None):
        self.left = left
        self.right = right
        self.depth = depth
        self.data = data
        self.seg = seg
        self.father = father

    def is_root(self):
        print('root:', self.father.__bool__())
        return not bool(self.father)

    def is_leave(self):
        print(self.left, self.right)
        return not(bool(self.left) or bool(self.right))

    def is_empty(self):
        return bool((self.depth == 0) or (self.data.shape[0] == 0))

    def append_left(self, left):
        self.left = left

    def append_right(self, right):
        self.right = right

    def set_father(self, father):
        self.father = father

    def all_data(self):
        #return np.concatenate((self.left_data(), self.data, self.right_data()), axis=0)
        return np.array(list(self.left_data()) + list(self.data) + list(self.right_data()))

    def left_data(self):
        if self.is_empty():
            return np.array([])
        else:
            return self.left.all_data()

    def right_data(self):
        if self.is_empty():
            return np.array([])
        else:
            return self.right.all_data()

    def find_leave_node(self, x):
        if self.is_leave():
            return self
        else:
            k = x.shape[0]
            j = self.depth % k
            if x[j] < self.seg:
                return self.left.find_leave_node(x)
            else:
                return self.right.find_leave_node(x)

    def search(self, x, k=1):
        x = np.array(x)
        # 维数
        km = x.shape[0]
        depth = self.depth
        # 坐标索引
        j = depth % km
        endleave = self.find_leave_node(x)
        N_k = np.zeros(k)
        d_k = np.zeros(k)
        node = endleave
        data = node.all_data()
        while len(data) < k:
            node = node.father
            data = node.all_data()
        ds = np.apply_along_axis(lambda xt: distance(x, xt), 1, data)
        dsargsort = ds.argsort()
        dsargsortargsort = dsargsort.argsort()
        d_k = ds[dsargsortargsort[:k]]
        N_k = data[dsargsortargsort[:k]]
        node = node.father
        while not node.is_root():
            for xt in node.data:
                N_k, d_k = self.__update_N_k(N_k, d_k, xt, x)
            for xt in node.right_data():
                N_k, d_k = self.__update_N_k(N_k, d_k, xt, x)
            node = node.father
        return N_k

    def __bool__(self):
        return not self.is_empty()

    @staticmethod
    def __update_N_k(N_k, d_k, xt, x):
        print('xt:', xt)
        ds = distance(xt, x)
        if d_k[-1] > ds:
            d_k[-1] = ds
            N_k[-1] = xt
        else:
            return N_k, d_k
        for i in range(len(d_k)-2, -1):
            if d_k[i] > ds:
                d_k[i+1] = d_k[i]
                N_k[i+1] = N_k[i]
                d_k[i] = ds
                N_k[i] = xt
            else:
                break
        return N_k, d_k


    @classmethod
    def __cons(cls, T, depth, k, father):
        if T.shape[0] == 0 or T is None:
            return kdTree()
        l = depth % k
        xlm = _median(T[:, l])
        leftT = T[T[:, l] < xlm]
        rightT = T[T[:, l] > xlm]
        data = T[T[:, l] == xlm]
        kT = cls(None, None, data, depth, xlm, father)
        lT = cls.__cons(leftT, depth+1, k, father=kT)
        rT = cls.__cons(rightT, depth+1, k, father=kT)
        kT.append_left(lT)
        kT.append_right(rT)
        return kT

    @classmethod
    def cons(cls, T):
        T = np.array(T)
        k = T.shape[1]
        depth = 0
        return cls.__cons(T, depth, k, None)

def _median(x):
    x = np.sort(x)
    return x[len(x)>>1]

def distance(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return ((x1 - x2)**2).sum()**0.5

def kNN_kd_tree_train(X, y, x, k):
    X = np.array(X)
    y = np.array(y)
    kT = kdTree.cons(X)
    N_k = kT.search(x, k)
    print(N_k)
    xindexs = np.array([(X == xk).all(axis=1) for xk in N_k])
    print(xindexs)
    yk = y[np.array(xindexs)]
    yunique = np.unique(yk)
    yend = yunique[(yk - yunique[:, np.newaxis]).sum(axis=1).argmax()]
    return yend

def test_kNN_kd_tree_train():
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
    ypredit = kNN_kd_tree_train(X, y, x, 3)
    if ypredit == -1:
        print('test kNN_kd_tree_train sucessfully')
    else:
        print('test kNN_kd_tree_train failed!')



def test_kdTree_cons():
    T = [
        [2, 3], [5, 4], [9, 6],
        [4, 7], [8,1], [7, 2]
    ]
    kdT = kdTree.cons(T)
    if all( [(kdT.data[0] == [7,2]).all() ,
            (kdT.left.data[0] == [5,4]).all(),
            (kdT.left.left.data[0] == [2, 3]).all(),
            (kdT.left.right.data[0] == [4,7]).all(),
            (kdT.right.data[0] == [9,6]).all(),
            (kdT.right.left.data[0] == [8, 1]).all()]
            ):
        print('test kdTree.cons successfully!')
    else:
        print('test kdTree.cons failed!')

def test_kdTree_search():
    T = [
        [2, 3], [5, 4], [9, 6],
        [4, 7], [8,1], [7, 2]
    ]
    kdT = kdTree.cons(T)
    N_k = kdT.search([3, 4.5])
    print(N_k)
    if (N_k[0] == [2, 3]).all():
        print('test kdTree.search successfully!')
    else:
        print('test kdTree.search failed!')

if __name__ == '__main__':
    test_kdTree_cons()
    test_kdTree_search()
    test_kNN_kd_tree_train()
