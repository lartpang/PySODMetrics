# -*- coding: utf-8 -*-
# @Time    : 2020/11/27
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import time

import numpy as np


# 快速统计numpy数组的非零值建议使用np.count_nonzero，一个简单的小实验
def cal_nonzero(size):
    a = np.random.randn(size, size)
    a = a > 0
    start = time.time()
    print(np.count_nonzero(a), time.time() - start)
    start = time.time()
    print(np.sum(a), time.time() - start)
    start = time.time()
    print(len(np.nonzero(a)[0]), time.time() - start)


if __name__ == '__main__':
    cal_nonzero(1000)
    # 499950 6.723403930664062e-05
    # 499950 0.0006949901580810547
    # 499950 0.007088184356689453
