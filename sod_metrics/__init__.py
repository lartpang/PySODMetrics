import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist

_EPS = 1e-16


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred.astype(np.float64)
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    else:
        if pred.max() > 1:
            pred = pred / 255
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)


class Fmeasure(object):
    # Fmeasure： Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        """
        gt is {0, 255} map and will be binarized by the threshold of 128
        pred is [0, 255] np.uint8
        """
        pred, gt = _prepare_data(pred, gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        快速统计numpy数组的非零值建议使用np.count_nonzero，一个简单的小实验
        def cal_nonzero(size):
        ...     a = np.random.randn(size, size)
        ...     a = a > 0
        ...     start = time.time()
        ...     print(np.count_nonzero(a), time.time() - start)
        ...     start = time.time()
        ...     print(np.sum(a), time.time() - start)
        ...     start = time.time()
        ...     print(len(np.nonzero(a)[0]), time.time() - start)
        ...
        cal_nonzero(1000)
        499792 6.67572021484375e-05
        499792 0.0006699562072753906
        499792 0.006061553955078125
        """
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
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
        # 1. 获取预测结果在真值前背景区域中的直方图
        bins = np.linspace(0, 256, 257) / 256
        fg_hist, _ = np.histogram(pred[gt], bins=bins)  # 最后一个bin为[255, 256]
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量, 这里仅计算了前景区域
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        # 3. 使用不同阈值的结果计算对应的precision和recall
        # p和r的计算的真值是pred==1&gt==1，二者仅有分母不同，分母前者是pred==1，后者是gt==1
        # 为了同时计算不同阈值的结果，这里使用hsitogram&flip&cumsum 获得了不同各自的前景像素数量
        TPs = fg_w_thrs.copy()
        Ps = (fg_w_thrs + bg_w_thrs).copy()
        T = np.count_nonzero(gt)
        # 为防止除0，这里针对除0的情况分析后直接对于0分母设为1，因为此时分子必为0
        Ps[Ps == 0] = 1
        if T == 0:
            T = 1
        # TODO: T=0 或者 特定阈值下fg_w_thrs=0或者bg_w_thrs=0，这些都会包含在TPs[i]=0的情况中，
        #  但是这里使用TPs不便于处理列表
        # T=0 -> fg_w_thrs=[0, ..., 0] -> TPs=[0, ..., 0] 解决办法：T重新赋值为1
        # Ps[i] = 0 -> fg_w_thrs[i] = 0, bg_w_thrs[i] = 0
        precisions = TPs / Ps
        recalls = TPs / T

        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, np.float64))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=np.float64), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=np.float64), axis=0)  # N, 256
        recall = np.mean(np.array(self.recalls, dtype=np.float64), axis=0)  # N, 256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm),
                    pr=dict(p=precision, r=recall))


class MAE(object):
    # mean absolute error
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        """
        gt is {0, 255} map and will be binarized by the threshold of 128
        pred is [0, 255] np.uint8
        """
        pred, gt = _prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> float:
        score = np.mean(np.abs(pred - gt))
        return score

    def get_results(self) -> dict:
        mae = np.mean(np.array(self.maes, np.float64))
        return dict(mae=mae)


class Smeasure(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + \
                    (1 - self.alpha) * self.region(pred, gt)
            score = max(0, score)
        return score

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1])
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        为了保证与matlab代码的一致性，这里对中心坐标进行了加一，在后面划分区域的时候就不用使用多余的加一操作
        因为matlab里的1:X生成的序列会包含X这个值
        """
        h, w = matrix.shape
        if matrix.sum() == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            area_object = np.sum(matrix)
            row_ids = np.arange(h)
            col_ids = np.arange(w)
            x = np.round(np.sum(np.sum(matrix, axis=0) * col_ids) / area_object)
            y = np.round(np.sum(np.sum(matrix, axis=1) * row_ids) / area_object)
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        # w4 = (h - y) * (w - x) / area
        w4 = 1 - w1 - w2 - w3

        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                    pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                    weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        sm = np.mean(np.array(self.sms, dtype=np.float64))
        return dict(sm=sm)


class Emeasure(object):
    # Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self, only_adaptive_em: bool = False):
        """
        Args:
            only_adaptive_em: 由于计算changeable耗时较长，为了用于模型的快速验证，可以选择不计算，仅保留adaptive_em
        """
        self.adaptive_ems = []
        self.changeable_ems = None if only_adaptive_em else []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)
        if self.changeable_ems is not None:
            changeable_ems = self.cal_changeable_em_light(pred, gt)
            self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        score = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return score

    def cal_changeable_em_light(self, pred: np.ndarray, gt: np.ndarray) -> list:
        changeable_ems = [
            self.cal_em_with_threshold(pred, gt, threshold=th)
            for th in np.linspace(0, 1, 256)
        ]
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        if np.all(~gt):
            enhanced_matrix = 1 - binarized_pred
        elif np.all(gt):
            enhanced_matrix = binarized_pred
        else:
            enhanced_matrix = self.cal_enhanced_matrix(binarized_pred, gt)
        score = enhanced_matrix.sum() / (gt.shape[0] * gt.shape[1] - 1 + _EPS)
        return score

    def cal_enhanced_matrix(self, dFM: np.ndarray, dGT: np.ndarray) -> np.ndarray:
        align_FM = dFM - dFM.mean()
        align_GT = dGT - dGT.mean()
        align_Matrix = 2.0 * (align_GT * align_FM) / (align_GT ** 2 + align_FM ** 2 + _EPS)
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    # def cal_changeable_em_fast(self, pred:np.ndarray, gt:np.ndarray):
    #     """
    #     会占用太大的内存，light是更合适的选择
    #     """
    #     binarized_preds = np.empty(shape=(256, *(gt.shape)), dtype=np.bool)
    #     for th in range(256):
    #         binarized_preds[th] = pred >= th
    #
    #     if self.all_bg:
    #         enhanced_matrix = 1 - binarized_preds
    #     elif self.all_fg:
    #         enhanced_matrix = binarized_preds
    #     else:
    #         enhanced_matrix = self.cal_enhanced_matrix_parallel(binarized_preds, gt)
    #     # N, H, W
    #     changeable_ems = enhanced_matrix.sum(axis=(1, 2)) / (self.gt_size - 1 + _EPS)
    #     # N
    #     return changeable_ems
    #
    # def cal_enhanced_matrix_parallel(self, dFM, dGT):
    #     """
    #     dFM: (N, H, W)
    #     dGT: (H, W)
    #     """
    #     align_FM = dFM - dFM.mean(axis=(1, 2), keepdims=True)
    #     align_GT = dGT - dGT.mean()  # H, W
    #     align_Matrix = 2.0 * (align_GT * align_FM) / (align_GT ** 2 + align_FM ** 2 + _EPS)
    #     enhanced = np.power(align_Matrix + 1, 2) / 4
    #     return enhanced

    def get_results(self) -> dict:
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=np.float64))
        if self.changeable_ems is not None:
            changeable_em = np.mean(np.array(self.changeable_ems, dtype=np.float64), axis=0)
        else:
            changeable_em = None
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFmeasure(object):
    """
    created by lartpang (Youwei Pang)
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
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
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=np.float64))
        return dict(wfm=weighted_fm)
