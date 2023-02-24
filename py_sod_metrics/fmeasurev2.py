# -*- coding: utf-8 -*-
import abc

import numpy as np

from .utils import TYPE, get_adaptive_threshold, prepare_data


class _BaseHandler:
    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
    ):
        """
        Args:
            with_dynamic (bool, optional): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool, optional): Record adaptive results for adp version.
            with_binary (bool, optional): Record binary results for binary version.
            sample_based (bool, optional): Whether to average the metric of each sample or calculate
                the metric of the dataset. Defaults to True.
        """
        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None
        self.sample_based = sample_based
        if with_binary:
            if self.sample_based:
                self.binary_results = []
            else:
                self.binary_results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        else:
            self.binary_results = None

    @abc.abstractmethod
    def __call__(self, *args, **kwds):
        pass

    @staticmethod
    def divide(numerator, denominator):
        denominator = np.array(denominator, dtype=TYPE)
        np.divide(numerator, denominator, out=denominator, where=denominator != 0)
        return denominator


class IOUHandler(_BaseHandler):
    """Intersection over Union

    iou = tp / (tp + fp + fn)
    """

    def __call__(self, tp, fp, tn, fn):
        # ious = np.where(Ps + FNs == 0, 0, TPs / (Ps + FNs))
        return self.divide(tp, tp + fp + fn)


class SpecificityHandler(_BaseHandler):
    """Specificity

    specificity = tn / (tn + fp)
    """

    def __call__(self, tp, fp, tn, fn):
        # specificities = np.where(TNs + FPs == 0, 0, TNs / (TNs + FPs))
        return self.divide(tn, tn + fp)


class DICEHandler(_BaseHandler):
    """DICE

    dice = 2 * tp / (tp + fn + tp + fp)
    """

    def __call__(self, tp, fp, tn, fn):
        # dices = np.where(TPs + FPs == 0, 0, 2 * TPs / (T + Ps))
        return self.divide(2 * tp, tp + fn + tp + fp)


class OverallAccuracyHandler(_BaseHandler):
    """OverallAccuracy

    oa = overall_accuracy = (tp + tn) / (tp + fp + tn + fn)
    """

    def __call__(self, tp, fp, tn, fn):
        # dices = np.where(TPs + FPs == 0, 0, 2 * TPs / (T + Ps))
        return self.divide(tp + tn, tp + fp + tn + fn)


class KappaHandler(_BaseHandler):
    """KappaAccuracy

    kappa = kappa = (oa - p_) / (1 - p_)
    p_ = [(tp + fp)(tp + fn) + (tn + fn)(tn + tp)] / (tp + fp + tn + fn)^2
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        beta: float = 0.3,
    ):
        """
        Args:
            with_dynamic (bool, optional): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool, optional): Record adaptive results for adp version.
            with_binary (bool, optional): Record binary results for binary version.
            sample_based (bool, optional): Whether to average the metric of each sample or calculate
                the metric of the dataset. Defaults to True.
            beta (bool, optional): β^2 in F-measure. Defaults to 0.3.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
        )

        self.beta = beta
        self.oa = OverallAccuracyHandler(False, False)

    def __call__(self, tp, fp, tn, fn):
        oa = self.oa(tp, fp, tn, fn)
        hpy_p = self.divide(
            (tp + fp) * (tp + fn) + (tn + fn) * (tn + tp),
            (tp + fp + tn + fn) ** 2,
        )
        return self.divide(oa - hpy_p, 1 - hpy_p)


class PrecisionHandler(_BaseHandler):
    """Precision

    precision = tp / (tp + fp)
    """

    def __call__(self, tp, fp, tn, fn):
        # precisions = np.where(Ps == 0, 0, TPs / Ps)
        return self.divide(tp, tp + fp)


class RecallHandler(_BaseHandler):
    """Recall

    recall = tp / (tp + fn)
    """

    def __call__(self, tp, fp, tn, fn):
        # recalls = np.where(TPs == 0, 0, TPs / T)
        return self.divide(tp, tp + fn)


class BERHandler(_BaseHandler):
    """Balance Error Rate

    ber = 1 - 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    """

    def __call__(self, tp, fp, tn, fn):
        fg = np.asarray(tp + fn, dtype=TYPE)
        bg = np.asarray(tn + fp, dtype=TYPE)
        np.divide(tp, fg, out=fg, where=fg != 0)
        np.divide(tn, bg, out=bg, where=bg != 0)
        return 1 - 0.5 * (fg + bg)


class FmeasureHandler(_BaseHandler):
    """F-measure

    fmeasure = (beta + 1) * precision * recall / (beta * precision + recall)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        beta: float = 0.3,
    ):
        """
        Args:
            with_dynamic (bool, optional): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool, optional): Record adaptive results for adp version.
            with_binary (bool, optional): Record binary results for binary version.
            sample_based (bool, optional): Whether to average the metric of each sample or calculate
                the metric of the dataset. Defaults to True.
            beta (bool, optional): β^2 in F-measure. Defaults to 0.3.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
        )

        self.beta = beta
        self.precision = PrecisionHandler(False, False)
        self.recall = RecallHandler(False, False)

    def __call__(self, tp, fp, tn, fn):
        # 为了对齐原始实现，这里不使用合并后的形式，仍然基于Precision和Recall的方式计算。
        # numerator = (self.beta + 1) * tp
        # denominator = (self.beta + 1) * tp + self.beta * fn + fp

        p = self.precision(tp, fp, tn, fn)
        r = self.recall(tp, fp, tn, fn)
        return self.divide((self.beta + 1) * p * r, self.beta * p + r)


class FmeasureV2:
    def __init__(self, metric_handlers: dict = None):
        """Enhanced Fmeasure class with more relevant metrics, e.g. precision, recall, specificity, dice, iou, fmeasure and so on.

        Args:
            metric_handlers (dict, optional): Handlers of different metrics. Defaults to None.
        """
        self._metric_handlers = metric_handlers if metric_handlers else {}

    def add_handler(self, handler_name, metric_handler):
        self._metric_handlers[handler_name] = metric_handler

    @staticmethod
    def get_statistics(binary: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        """Calculate the TP, FP, TN and FN based a adaptive threshold.

        Args:
            binary (np.ndarray): binarized `pred` containing [0, 1]
            gt (np.ndarray): gt binarized by 128
            FG (int): the number of foreground pixels in gt
            BG (int): the number of background pixels in gt

        Returns:
            dict: TP, FP, TN, FN
        """
        TP = np.count_nonzero(binary[gt])
        FP = np.count_nonzero(binary[~gt])
        FN = FG - TP
        TN = BG - FP
        return {"tp": TP, "fp": FP, "tn": TN, "fn": FN}

    def adaptively_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        """Calculate the TP, FP, TN and FN based a adaptive threshold.

        Args:
            pred (np.ndarray): prediction normalized in [0, 1]
            gt (np.ndarray): gt binarized by 128
            FG (int): the number of foreground pixels in gt
            BG (int): the number of background pixels in gt

        Returns:
            dict: TP, FP, TN, FN
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary = pred >= adaptive_threshold
        return self.get_statistics(binary, gt, FG, BG)

    def dynamically_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        """Calculate the corresponding TP, FP, TN and FNs when the threshold changes from 0 to 255.

        Args:
            pred (np.ndarray): prediction normalized in [0, 1]
            gt (np.ndarray): gt binarized by 128
            FG (int): the number of foreground pixels in gt
            BG (int): the number of background pixels in gt

        Returns:
            dict: TPs, FPs, TNs, FNs
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
        return {"tp": TPs, "fp": FPs, "tn": TNs, "fn": FNs}

    def step(self, pred: np.ndarray, gt: np.ndarray):
        """Statistics the metrics for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.

        Raises:
            ValueError: Please add your metric handler before using `step()`.
        """
        if not self._metric_handlers:  # 没有添加metric_handler
            raise ValueError("Please add your metric handler before using `step()`.")

        pred, gt = prepare_data(pred, gt)

        FG = np.count_nonzero(gt)  # 真实前景, FG=(TPs+FNs)
        BG = gt.size - FG  # 真实背景, BG=(TNs+FPs)

        dynamical_tpfptnfn = None
        adaptive_tpfptnfn = None
        binary_tpfptnfn = None
        for handler_name, handler in self._metric_handlers.items():
            if handler.dynamic_results is not None:
                if dynamical_tpfptnfn is None:
                    dynamical_tpfptnfn = self.dynamically_binarizing(
                        pred=pred, gt=gt, FG=FG, BG=BG
                    )
                handler.dynamic_results.append(handler(**dynamical_tpfptnfn))

            if handler.adaptive_results is not None:
                if adaptive_tpfptnfn is None:
                    adaptive_tpfptnfn = self.adaptively_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)
                handler.adaptive_results.append(handler(**adaptive_tpfptnfn))

            if handler.binary_results is not None:
                if binary_tpfptnfn is None:
                    # `pred > 0.5`: Simulating the effect of the `argmax` function.
                    binary_tpfptnfn = self.get_statistics(binary=pred > 0.5, gt=gt, FG=FG, BG=BG)
                if handler.sample_based:
                    handler.binary_results.append(handler(**binary_tpfptnfn))
                else:
                    handler.binary_results["tp"] += binary_tpfptnfn["tp"]
                    handler.binary_results["fp"] += binary_tpfptnfn["fp"]
                    handler.binary_results["tn"] += binary_tpfptnfn["tn"]
                    handler.binary_results["fn"] += binary_tpfptnfn["fn"]

    def get_results(self) -> dict:
        """Return the results of the specific metric names.

        Returns:
            dict: All results corresponding to different metrics.
        """
        results = {}
        for handler_name, handler in self._metric_handlers.items():
            res = {}
            if handler.dynamic_results is not None:
                res["dynamic"] = np.mean(np.array(handler.dynamic_results, dtype=TYPE), axis=0)
            if handler.adaptive_results is not None:
                res["adaptive"] = np.mean(np.array(handler.adaptive_results, dtype=TYPE))
            if handler.binary_results is not None:
                if handler.sample_based:
                    res["binary"] = np.mean(np.array(handler.binary_results, dtype=TYPE))
                else:
                    # NOTE: use `np.mean` to simplify output format (`array(123)` -> `123`)
                    res["binary"] = np.mean(handler(**handler.binary_results))
            results[handler_name] = res
        return results
