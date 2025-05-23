# -*- coding: utf-8 -*-
import warnings

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

from .utils import EPS, TYPE, get_adaptive_threshold, validate_and_normalize_input


class Fmeasure(object):
    def __init__(self, beta: float = 0.3):
        """F-measure for SOD.

        ```
        @inproceedings{Fmeasure,
            title={Frequency-tuned salient region detection},
            author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
            booktitle=CVPR,
            number={CONF},
            pages={1597--1604},
            year={2009}
        }
        ```

        Args:
            beta (float): the weight of the precision
        """
        warnings.warn("This class will be removed in the future, please use FmeasureV2 instead!")

        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the adaptive F-measure.

        Returns:
            float: adaptive_fm
        """
        # ``np.count_nonzero`` is faster and better
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        """Calculate the corresponding precision and recall when the threshold changes from 0 to 255.

        These precisions and recalls can be used to obtain the mean F-measure, maximum F-measure,
        precision-recall curve and F-measure-threshold curve.

        For convenience, `changeable_fms` is provided here, which can be used directly to obtain
        the mean F-measure, maximum F-measure and F-measure-threshold curve.

        Returns:
            tuple: (precisions, recalls, changeable_fms)
        """
        # 1. 获取预测结果在真值前背景区域中的直方图
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)  # 最后一个bin为[255, 256]
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量, 这里仅计算了前景区域
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        # 3. 使用不同阈值的结果计算对应的precision和recall
        # p和r的计算的真值是pred==1&gt==1，二者仅有分母不同，分母前者是pred==1，后者是gt==1
        # 为了同时计算不同阈值的结果，这里使用hsitogram&flip&cumsum 获得了不同各自的前景像素数量
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        # 为防止除0，这里针对除0的情况分析后直接对于0分母设为1，因为此时分子必为0
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)
        # TODO: T=0 或者 特定阈值下fg_w_thrs=0或者bg_w_thrs=0，这些都会包含在TPs[i]=0的情况中，
        #  但是这里使用TPs不便于处理列表
        precisions = TPs / Ps
        recalls = TPs / T

        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        """Return the results about F-measure.

        Returns:
            dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))
        """
        adaptive_fm = np.mean(np.array(self.adaptive_fms, TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=TYPE), axis=0)  # N, 256
        recall = np.mean(np.array(self.recalls, dtype=TYPE), axis=0)  # N, 256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))


class MAE(object):
    def __init__(self):
        """MAE(mean absolute error) for SOD.

        ```
        @inproceedings{MAE,
            title={Saliency filters: Contrast based filtering for salient region detection},
            author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
            booktitle=CVPR,
            pages={733--740},
            year={2012}
        }
        ```
        """
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Calculate the mean absolute error.

        Returns:
            np.ndarray: mae
        """
        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        """Return the results about MAE.

        Returns:
            dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, TYPE))
        return dict(mae=mae)


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        """S-measure(Structure-measure) of SOD.

        ```
        @inproceedings{Smeasure,
            title={Structure-measure: A new way to eval foreground maps},
            author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
            booktitle=ICCV,
            pages={4548--4557},
            year={2017}
        }
        ```

        Args:
            alpha: the weight for balancing the object score and the region score
        """
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the S-measure.

        Returns:
            s-measure
        """
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            object_score = self.object(pred, gt) * self.alpha
            region_score = self.region(pred, gt) * (1 - self.alpha)
            sm = max(0, object_score + region_score)
        return sm

    def s_object(self, x: np.ndarray) -> float:
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        score = 2 * mean / (np.power(mean, 2) + 1 + std + EPS)
        return score

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the object score."""
        gt_mean = np.mean(gt)
        fg_score = self.s_object(pred[gt]) * gt_mean
        bg_score = self.s_object((1 - pred)[~gt]) * (1 - gt_mean)
        object_score = fg_score + bg_score
        return object_score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the region score."""
        h, w = gt.shape
        area = h * w

        # Calculate the centroid coordinate of the foreground
        if np.count_nonzero(gt) == 0:
            cy, cx = np.round(h / 2), np.round(w / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            cy, cx = np.argwhere(gt).mean(axis=0).round()
        # To ensure consistency with the matlab code, one is added to the centroid coordinate,
        # so there is no need to use the redundant addition operation when dividing the region later,
        # because the sequence generated by ``1:X`` in matlab will contain ``X``.
        cy, cx = int(cy) + 1, int(cx) + 1

        # Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        w_lt = cx * cy / area
        w_rt = cy * (w - cx) / area
        w_lb = (h - cy) * cx / area
        w_rb = 1 - w_lt - w_rt - w_lb
        score_lt = self.ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) * w_lt
        score_rt = self.ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) * w_rt
        score_lb = self.ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) * w_lb
        score_rb = self.ssim(pred[cy:h, cx:w], gt[cy:h, cx:w]) * w_rb
        return score_lt + score_rt + score_lb + score_rb

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the ssim score."""
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        """Return the results about S-measure.

        Returns:
            dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=TYPE))
        return dict(sm=sm)


class Emeasure(object):
    def __init__(self):
        """E-measure(Enhanced-alignment Measure) for SOD.

        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38

        ```
        @inproceedings{Emeasure,
            title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
            author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
            booktitle=IJCAI,
            pages="698--704",
            year={2018}
        }
        ```
        """
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the adaptive E-measure.

        Returns:
            adaptive_em
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Calculate the changeable E-measure, which can be used to obtain the mean E-measure, the maximum E-measure and the E-measure-threshold curve.

        Returns:
            changeable_ems
        """
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        """Calculate the E-measure corresponding to the specific threshold.

        Variable naming rules within the function:
        `[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]`

        If only `pred` or `gt` is considered, another corresponding attribute location is replaced with '`_`'.
        """
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Calculate the E-measure corresponding to the threshold that varies from 0 to 255..

        Variable naming rules within the function:
        `[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]`

        If only `pred` or `gt` is considered, another corresponding attribute location is replaced with '`_`'.
        """
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
        return em

    def generate_parts_numel_combinations(
        self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
    ):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        """Return the results about E-measure.

        Returns:
            dict(em=dict(adp=adaptive_em, curve=changeable_em))
        """
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFmeasure(object):
    def __init__(self, beta: float = 1):
        """Weighted F-measure for SOD.

        ```
        @inproceedings{wFmeasure,
            title={How to eval foreground maps?},
            author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
            booktitle=CVPR,
            pages={248--255},
            year={2014}
        }
        ```

        Args:
            beta (float): the weight of the precision
        """
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the weighted F-measure."""
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        # 注意这里使用mask索引矩阵的时候不可使用Ew[gt]，这实际上仅在索引Ew的0维度
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """2D gaussian mask - should give the same result as MATLAB's:
        `fspecial('gaussian',[shape],[sigma])`
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        """Return the results about weighted F-measure.

        Returns:
            dict(wfm=weighted_fm)
        """
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=TYPE))
        return dict(wfm=weighted_fm)
