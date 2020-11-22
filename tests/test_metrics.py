# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

import cv2

import sod_metrics as M

FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()

data_root = './test_data'
mask_root = os.path.join(data_root, 'masks')
pred_root = os.path.join(data_root, 'preds')
for mask_name in os.listdir(mask_root):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

fm = FM.get_results()['fm']
wfm = WFM.get_results()['wfm']
sm = SM.get_results()['sm']
em = EM.get_results()['em']
mae = MAE.get_results()['mae']

print(
    'Smeasure', sm.round(3),
    'wFmeasure', wfm.round(3),
    'MAE', mae.round(3),
    'adpEm', em['adp'].round(3),
    'meanEm', em['curve'].mean().round(3),
    'maxEm', em['curve'].max().round(3),
    'adpFm', fm['adp'].round(3),
    'meanFm', fm['curve'].mean().round(3),
    'maxFm', fm['curve'].max().round(3),
)

# ours:   Smeasure 0.959  wFmeasure 0.438  MAE 0.018  adpEm 0.946  meanEm 0.975  maxEm 0.987
# adpFm 0.456  meanFm 0.454  maxFm 0.461
# matlab: Smeasure:0.959; wFmeasure:0.438; MAE:0.018; adpEm:0.946; meanEm:0.977; maxEm:0.987;
# adpFm:0.456; meanFm:0.454; maxFm:0.461.
