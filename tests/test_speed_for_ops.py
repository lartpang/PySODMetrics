# -*- coding: utf-8 -*-
# @Time    : 2020/11/27
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import time

import numpy as np


# TPs = fg_w_thrs.copy()
# Ps = (fg_w_thrs + bg_w_thrs).copy()
# T = np.count_nonzero(gt)
# Ps[Ps == 0] = 1
# if T == 0:
#     T = 1
# 0.06196483850479126
def ori_process_for_copyop(fg_w_thrs, bg_w_thrs, gt):
    TPs = fg_w_thrs.copy()
    Ps = (fg_w_thrs + bg_w_thrs).copy()
    T = np.count_nonzero(gt)
    Ps[Ps == 0] = 1
    if T == 0:
        T = 1


# 0.060863380432128904
def new_process_for_copyop(fg_w_thrs, bg_w_thrs, gt):
    TPs = fg_w_thrs
    Ps = fg_w_thrs + bg_w_thrs
    T = max(np.count_nonzero(gt), 1)
    Ps[Ps == 0] = 1


if __name__ == '__main__':
    size = 1000
    fg_w_thrs = np.random.randn(size, size) > 0
    bg_w_thrs = np.random.randn(size, size) < 0
    gt = np.random.randn(size, size) > 0
    start = time.time()
    for _ in range(1000):
        ori_process_for_copyop(fg_w_thrs=fg_w_thrs, bg_w_thrs=bg_w_thrs, gt=gt)
    print((time.time() - start) / 100)
    start = time.time()
    for _ in range(1000):
        new_process_for_copyop(fg_w_thrs=fg_w_thrs, bg_w_thrs=bg_w_thrs, gt=gt)
    print((time.time() - start) / 100)
    # 0.042382607460021975
    # 0.040832839012145995
