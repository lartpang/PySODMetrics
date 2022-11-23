# -*- coding: utf-8 -*-
import numpy as np

from .utils import TYPE, get_adaptive_threshold, prepare_data


class IOUHandler:
    name = "iou"

    def __init__(self, with_dynamic: bool = True, with_adaptive: bool = True):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None

    def __call__(self, tp, fp, tn, fn):
        # ious = np.where(Ps + FNs == 0, 0, TPs / (Ps + FNs))
        numerator = tp
        denominator = tp + fp + fn
        return np.divide(numerator, denominator, where=denominator != 0, dtype=TYPE)


class SpecificityHandler:
    name = "specificity"

    def __init__(self, with_dynamic: bool = True, with_adaptive: bool = True):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None

    def __call__(self, tp, fp, tn, fn):
        # specificities = np.where(TNs + FPs == 0, 0, TNs / (TNs + FPs))
        numerator = tn
        denominator = tn + fp
        return np.divide(numerator, denominator, where=denominator != 0, dtype=TYPE)


class DICEHandler:
    name = "dice"

    def __init__(self, with_dynamic: bool = True, with_adaptive: bool = True):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None

    def __call__(self, tp, fp, tn, fn):
        # dices = np.where(TPs + FPs == 0, 0, 2 * TPs / (T + Ps))
        numerator = 2 * tp
        denominator = tp + fn + tp + fp
        return np.divide(numerator, denominator, where=denominator != 0, dtype=TYPE)


class PrecisionHandler:
    name = "precision"

    def __init__(self, with_dynamic: bool = True, with_adaptive: bool = True):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None

    def __call__(self, tp, fp, tn, fn):
        # precisions = np.where(Ps == 0, 0, TPs / Ps)
        numerator = tp
        denominator = tp + fp
        return np.divide(numerator, denominator, where=denominator != 0, dtype=TYPE)


class RecallHandler:
    name = "recall"

    def __init__(self, with_dynamic: bool = True, with_adaptive: bool = True):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None

    def __call__(self, tp, fp, tn, fn):
        # recalls = np.where(TPs == 0, 0, TPs / T)
        numerator = tp
        denominator = tp + fn
        return np.divide(numerator, denominator, where=denominator != 0, dtype=TYPE)


class FmeasureHandler:
    name = "fmeasure"

    def __init__(self, with_dynamic: bool = True, with_adaptive: bool = True, beta: float = 0.3):
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None
        self.beta = beta
        self.precision = PrecisionHandler(False, False)
        self.recall = RecallHandler(False, False)

    def __call__(self, tp, fp, tn, fn):
        # 为了对齐原始实现，这里不使用合并后的形式，仍然基于Precision和Recall的方式计算。
        # numerator = (self.beta + 1) * tp
        # denominator = (self.beta + 1) * tp + self.beta * fn + fp

        p = self.precision(tp, fp, tn, fn)
        r = self.recall(tp, fp, tn, fn)
        numerator = (self.beta + 1) * p * r
        denominator = self.beta * p + r
        return np.divide(numerator, denominator, where=denominator != 0, dtype=TYPE)


class FmeasureV2:
    def __init__(self, metric_handlers: list = None):
        """Enhanced Fmeasure class with more relevant metrics, e.g. precision, recall, specificity, dice, iou, and fmeasure.

        Args:
            metric_handlers (list, optional): Handlers of different metrics. Defaults to None.
        """
        self._metric_handlers = metric_handlers if metric_handlers else []

    def add_handler(self, metric_handler):
        self._metric_handlers.append(metric_handler)

    def adaptively_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> tuple:
        """Calculate the TP, FP, TN and FN based a adaptive threshold.

        Args:
            pred (np.ndarray): prediction normalized in [0, 1]
            gt (np.ndarray): gt binarized by 128
            FG (int): the number of foreground pixels in gt
            BG (int): the number of background pixels in gt

        Returns:
            float: TP, FP, TN, FN
        """
        # ``np.count_nonzero`` is faster and better
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        TP = np.count_nonzero(binary_predcition[gt])
        FP = np.count_nonzero(binary_predcition[~gt])
        FN = FG - TP
        TN = BG - FP
        return TP, FP, TN, FN

    def dynamically_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> tuple:
        """Calculate the corresponding TP, FP, TN and FNs when the threshold changes from 0 to 255.

        Args:
            pred (np.ndarray): prediction normalized in [0, 1]
            gt (np.ndarray): gt binarized by 128
            FG (int): the number of foreground pixels in gt
            BG (int): the number of background pixels in gt

        Returns:
            tuple: TPs, FPs, TNs, FNs
        """
        # 1. 获取预测结果在真值前背景区域中的直方图
        pred: np.ndarray = (pred * 255).astype(np.uint8)
        bins: np.ndarray = np.linspace(0, 256, 257)
        tp_hist, _ = np.histogram(pred[gt], bins=bins)  # 最后一个bin为[255, 256]
        fp_hist, _ = np.histogram(pred[~gt], bins=bins)

        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量, 这里仅计算了前景区域
        tp_w_thrs = np.cumsum(np.flip(tp_hist))  # >= 255, >= 254, ... >= 1, >= 0
        fp_w_thrs = np.cumsum(np.flip(fp_hist))

        # 3. 计算对应的TP,FP,TN,FN
        TPs = tp_w_thrs  # 前景 预测为 前景
        FPs = fp_w_thrs  # 背景 预测为 前景
        FNs = FG - TPs  # 前景 预测为 背景
        TNs = BG - FPs  # 背景 预测为 背景
        return TPs, FPs, TNs, FNs

    def step(self, pred: np.ndarray, gt: np.ndarray):
        if not self._metric_handlers:  # 没有添加metric_handler
            raise ValueError("Please add your metric handler before using `step()`.")

        pred, gt = prepare_data(pred, gt)

        FG = np.count_nonzero(gt)  # 真实前景, FG=(TPs+FNs)
        BG = gt.size - FG  # 真实背景, BG=(TNs+FPs)
        TPs, FPs, TNs, FNs = self.dynamically_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)
        TP, FP, TN, FN = self.adaptively_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)

        for handler in self._metric_handlers:
            if handler.dynamic_results is not None:
                handler.dynamic_results.append(handler(tp=TPs, fp=FPs, tn=TNs, fn=FNs))
            if handler.adaptive_results is not None:
                handler.adaptive_results.append(handler(tp=TP, fp=FP, tn=TN, fn=FN))

    def get_results(self) -> dict:
        """Return the results of the specific metric names.

        Returns:
            dict: All results corresponding to different metrics.
        """
        results = {}
        for handler in self._metric_handlers:
            res = {}
            if handler.dynamic_results is not None:
                res["dynamic"] = np.mean(np.array(handler.dynamic_results, dtype=TYPE), axis=0)
            if handler.adaptive_results is not None:
                res["adaptive"] = np.mean(np.array(handler.adaptive_results, dtype=TYPE))
            results[handler.name] = res
        return results
