# -*- coding: utf-8 -*-
# @Time    : 2021/3/5
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang


import numpy as np

from py_sod_metrics.sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure


class CalTotalMetric(object):
    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape, gt_path)
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }
        results = {name: metric.round(bit_width) for name, metric in results.items()}
        return results


if __name__ == '__main__':
    cal_total_seg_metrics = CalTotalMetric()
    for batch in data_loader:
        seg_preds = model(batch)
        for seg_pred in seg_preds:
            cal_total_seg_metrics.step(seg_pred, mask_array, mask_path)
    fixed_seg_results = cal_total_seg_metrics.get_results()
