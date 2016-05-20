import math

_exp = math.exp

_sqrt = math.sqrt

import random

_random = random.random

# 模拟退火算法
def simulated_anneal(J, p, get_next, T=100, delta=0.98, T_min=1e-8):
    """
    J: 目标函数
    p: 函数的参数, 要实现deepcopy方法
    get_next: 参数的更新函数, 返回值和参数值不能共用同一个引用
    T: 初始温度
    delta: 降温速度
    T_min: 终止条件
    """
    J_pre = J(p)
    curp = p
    t = T
    # J_min 和 curp是保存临时最优解的地方
    J_min = J_pre
    minp = curp
    while t > T_min:
        newp = get_next(curp)
        J_new = J(newp)
        dE = J_new - J_pre
        if dE <= 0: # 得到更优解, 更新之
            curp = newp
            J_pre = J_new
            if J_new < J_min:
                J_min = J_new
                minp = curp
        elif (_exp(-dE/t) > _random()):# 以一定概率新的搜索
            curp = newp
            J_pre = J_new
        t *= delta
    return J_min, minp

# TSP问题的求解

class Path(list):
    def deepcopy(self):
        cp = Path(City(0,0,0) for i in range(self.__len__()))
        cp[:] = self[:]
        return cp

    @property
    def length(self):
        lth = 0
        for i in range(self.__len__()):
            lth += dist(self.__getitem__(i), self.__getitem__(i-1))
        return lth

from collections import namedtuple

City = namedtuple('City', ['i', 'x', 'y'])

def dist(pa, pb):
    return _sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2)

def GetNextPath(pth):
    n = len(pth)
    x = int((n-1) * _random()) + 1
    y = int((n-1) * _random()) + 1
    while x == y:
         x = int((n-1) * _random()) + 1
         y = int((n-1) * _random()) + 1
    # swap, 也是用随机方法产生新的路径
    x, y = (x, y) if x < y else (y, x)
    nxtpth = Path(City(0, 0, 0) for i in range(n))
    nxtpth[:x] = pth[:x]
    # nxtpth[x], nxtpth[y] = pth[y], pth[x]
    for t in range(x, y+1):
        nxtpth[t] = pth[x+y-t]
    nxtpth[y+1:] = pth[y+1:]
    return nxtpth
    #for t in range(x, (x+y+1)>>1):
    #    pth[t], pth[x+y-t] = pth[x+y-t], pth[t]
    #return pth

def TSP(pth):
    ipth = pth
    random.shuffle(ipth[1:])
    J = lambda p: p.length
    return simulated_anneal(J, ipth, GetNextPath, delta=0.995)

def test_JSP():
    citys = [(41, 94), (37, 84), (53, 67), (25, 62), (7, 64), (2, 99), (68, 58), (71, 44), (54, 62), (83, 69), (64, 60), (18, 54), (22, 60), (83, 46), (91, 38), (25, 38), (24, 42), (58, 69), (71, 71), (74, 78), (87, 76), (18, 40), (13, 40), (82, 7), (62, 32), (58, 35), (45, 21)]
    pth = Path(City(i, *it) for i, it in enumerate(citys))
    ml, pt_path = TSP(pth)
    print(pt_path.length, ml)
    print(pt_path)

if __name__ == '__main__':
    test_JSP()
