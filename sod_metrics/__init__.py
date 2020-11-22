import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist

_EPS = 1e-16


class Fmeasure(object):
    # Fmeasure： Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, beta=0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []

    def step(self, pred, gt):
        """
        gt is {0, 255} map and will be binarized by the threshold of 128
        pred is [0, 255] np.uint8
        """
        pred, gt = self.prepare_data(pred, gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)

    def prepare_data(self, pred, gt):
        gt = gt > 128
        pred = pred.astype(np.float32)
        return pred, gt

    def cal_adaptive_fm(self, pred, gt):
        adaptive_threshold = min(2 * pred.mean(), 255)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = np.sum(binary_predcition * gt)
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.sum(binary_predcition)
            rec = area_intersection / np.sum(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred, gt):
        # 1. 获取预测结果在真值前背景区域中的直方图
        fg_hist, _ = np.histogram(pred[gt == 1], bins=range(257))  # 最后一个bin为[255, 256]
        bg_hist, _ = np.histogram(pred[gt == 0], bins=range(257))
        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量,
        # yinwei zhebufen shi jisuan huiyongdao de qianjingqvyu
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        # 3. 使用不同阈值的结果计算对应的precision和recall
        # p和r的计算的真值是pred==1&gt==1，二者仅有分母不同，分母前者是pred==1，后者是gt==1
        # weile tongshi jisuan butong yuzhi de jieguo, zheli shiyong hsitogram&flop&cumsum
        # huodele butongyuzhixia de qianjing xiangsushuliang
        TPs = fg_w_thrs.copy()
        Ps = (fg_w_thrs + bg_w_thrs).copy()
        T = np.sum(gt)
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
        return precisions, recalls

    def get_results(self):
        adaptive_fm = np.mean(np.array(self.adaptive_fms, np.float32))

        precision = np.mean(np.array(self.precisions, dtype=np.float32), axis=0)
        recall = np.mean(np.array(self.recalls, dtype=np.float32), axis=0)

        numerator = (1 + self.beta) * precision * recall
        denominator = np.where(numerator == 0, 1, self.beta * precision + recall)
        changable_fm = numerator / denominator

        return dict(fm=dict(adp=adaptive_fm, curve=changable_fm),
                    pr=dict(p=precision, r=recall))


class MAE(object):
    # mean absolute error
    def __init__(self):
        self.maes = []

    def step(self, pred, gt):
        """
        gt is {0, 255} map and will be binarized by the threshold of 128
        pred is [0, 255] np.uint8
        """
        pred, gt = self.prepare_data(pred, gt)

        score = self.cal_mae(pred, gt)
        self.maes.append(score)

    def prepare_data(self, pred, gt):
        gt = gt > 128
        pred = pred.astype(np.float32)
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            pred /= 255
        return pred, gt

    def cal_mae(self, pred, gt):
        return np.mean(np.abs(pred - gt))

    def get_results(self):
        mae = np.mean(np.array(self.maes, np.float32))
        return dict(mae=mae)


class Smeasure(object):
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred, gt):
        pred, gt = self.prepare_data(pred=pred, gt=gt)
        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def prepare_data(self, pred, gt):
        gt = gt > 128
        pred = pred.astype(np.float32)
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            pred /= 255
        return pred, gt

    def cal_sm(self, pred, gt):
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

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, prediction, gt):
        x = np.mean(prediction[gt == 1])
        sigma_x = np.std(prediction[gt == 1])
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred, gt):
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

    def centroid(self, gt):
        """
        为了保证与matlab代码的一致性，这里对中心坐标进行了加一，在后面划分区域的时候就不用使用多余的加一操作
        因为matlab里的1:X生成的序列会包含X这个值
        """
        h, w = gt.shape
        if gt.sum() == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            area_object = np.sum(gt)
            row_ids = np.arange(h)
            col_ids = np.arange(w)
            x = np.round(np.sum(np.sum(gt, axis=0) * col_ids) / area_object)
            y = np.round(np.sum(np.sum(gt, axis=1) * row_ids) / area_object)
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred, gt, x, y):
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

    def ssim(self, pred, gt):
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

    def get_results(self):
        sm = np.mean(np.array(self.sms, dtype=np.float32))
        return dict(sm=sm)


class Emeasure(object):
    # Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    def __init__(self):
        self.adaptive_ems = []
        self.changeble_ems = []

    def step(self, pred, gt):
        pred, gt = self.prepare_data(pred=pred, gt=gt)
        self.set_shared_attr(pred=pred, gt=gt)
        changable_ems = self.cal_changable_em_light(pred, gt)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.changeble_ems.append(changable_ems)
        self.adaptive_ems.append(adaptive_em)

    def prepare_data(self, pred, gt):
        gt = gt > 128
        gt = gt.astype(np.float32)
        pred = pred.astype(np.float32)
        return pred, gt

    def set_shared_attr(self, pred, gt):
        self.all_fg = np.all(gt == 1)
        self.all_bg = np.all(gt == 0)
        self.gt_size = gt.shape[0] * gt.shape[1]

    def cal_adaptive_em(self, pred, gt):
        th = min(2 * pred.mean(), 255)
        binarized_pred = pred >= th

        if self.all_bg:
            enhanced_matrix = 1 - binarized_pred
        elif self.all_fg:
            enhanced_matrix = binarized_pred
        else:
            enhanced_matrix = self.cal_enhanced_matrix(binarized_pred, gt)
        score = np.sum(enhanced_matrix) / (self.gt_size - 1 + _EPS)
        return score

    def cal_changable_em_light(self, pred, gt):
        changable_ems = []
        for th in range(256):
            binarized_pred = pred >= th

            if self.all_bg:
                enhanced_matrix = 1 - binarized_pred
            elif self.all_fg:
                enhanced_matrix = binarized_pred
            else:
                enhanced_matrix = self.cal_enhanced_matrix(binarized_pred, gt)
            changable_em = enhanced_matrix.sum() / (self.gt_size - 1 + _EPS)
            changable_ems.append(changable_em)
        return changable_ems

    def cal_enhanced_matrix(self, dFM, dGT):
        """
        dFM: H, W
        dGT: H, W
        """
        align_FM = dFM - dFM.mean()
        align_GT = dGT - dGT.mean()
        align_Matrix = 2.0 * (align_GT * align_FM) / (align_GT ** 2 + align_FM ** 2 + _EPS)
        enhanced = np.power(align_Matrix + 1, 2) / 4
        return enhanced

    # def cal_changable_em_fast(self, pred, gt):
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
    #     changable_ems = enhanced_matrix.sum(axis=(1, 2)) / (self.gt_size - 1 + _EPS)
    #     # N
    #     return changable_ems
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

    def get_results(self):
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=np.float32))
        changable_em = np.mean(np.array(self.changeble_ems, dtype=np.float32), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changable_em))


class WeightedFmeasure(object):
    """
    created by lartpang (Youwei Pang)
    """

    def __init__(self, beta=1):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred, gt):
        pred, gt = self.prepare_data(pred=pred, gt=gt)

        if gt.max() == 0:
            score = 0
        else:
            score = self.cal_wfm(pred, gt)
        self.weighted_fms.append(score)

    def prepare_data(self, pred, gt):
        gt = gt > 128
        if pred.max() != pred.min():
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            pred /= 255
        return pred, gt

    def cal_wfm(self, pred, gt):
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

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
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

    def get_results(self):
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=np.float32))
        return dict(wfm=weighted_fm)
