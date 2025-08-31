import os
import sys

import cv2
import numpy as np

sys.path.append("..")
import py_sod_metrics


def ndarray_to_basetype(data):
    """
    将单独的ndarray，或者tuple，list或者dict中的ndarray转化为基本数据类型，
    即列表(.tolist())和python标量
    """

    def _to_list_or_scalar(item):
        listed_item = item.tolist()
        if isinstance(listed_item, list) and len(listed_item) == 1:
            listed_item = listed_item[0]
        return listed_item

    if isinstance(data, (tuple, list)):
        results = [_to_list_or_scalar(item) for item in data]
    elif isinstance(data, dict):
        results = {k: _to_list_or_scalar(item) for k, item in data.items()}
    else:
        assert isinstance(data, np.ndarray)
        results = _to_list_or_scalar(data)
    return results


INDIVADUAL_METRIC_MAPPING = {
    "mae": py_sod_metrics.MAE,
    "fm": py_sod_metrics.Fmeasure,
    "em": py_sod_metrics.Emeasure,
    "sm": py_sod_metrics.Smeasure,
    "wfm": py_sod_metrics.WeightedFmeasure,
    "hce": py_sod_metrics.HumanCorrectionEffortMeasure,
}


class GrayscaleMetricRecorderV1:
    def __init__(self):
        """
        用于统计各种指标的类
        https://github.com/lartpang/Py-SOD-VOS-EvalToolkit/blob/81ce89da6813fdd3e22e3f20e3a09fe1e4a1a87c/utils/recorders/metric_recorder.py

        主要应用于旧版本实现中的五个指标，即mae/fm/sm/em/wfm/hce。推荐使用V2版本。
        """
        self.mae = INDIVADUAL_METRIC_MAPPING["mae"]()
        self.fm = INDIVADUAL_METRIC_MAPPING["fm"]()
        self.sm = INDIVADUAL_METRIC_MAPPING["sm"]()
        self.em = INDIVADUAL_METRIC_MAPPING["em"]()
        self.wfm = INDIVADUAL_METRIC_MAPPING["wfm"]()
        self.hce = INDIVADUAL_METRIC_MAPPING["hce"]()

    def step(self, pre: np.ndarray, gt: np.ndarray):
        assert pre.shape == gt.shape
        assert pre.dtype == np.uint8
        assert gt.dtype == np.uint8

        self.mae.step(pre, gt)
        self.sm.step(pre, gt)
        self.fm.step(pre, gt)
        self.em.step(pre, gt)
        self.wfm.step(pre, gt)
        self.hce.step(pre, gt)

    def get_results(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        """
        返回指标计算结果：

        - 曲线数据(sequential)： fm/em/p/r
        - 数值指标(numerical)： SM/MAE/maxE/avgE/adpE/maxF/avgF/adpF/wFm
        """
        fm_info = self.fm.get_results()
        fm = fm_info["fm"]
        pr = fm_info["pr"]
        wfm = self.wfm.get_results()["wfm"]
        sm = self.sm.get_results()["sm"]
        em = self.em.get_results()["em"]
        mae = self.mae.get_results()["mae"]
        hce = self.hce.get_results()["hce"]

        sequential_results = {
            "fm": np.flip(fm["curve"]),
            "em": np.flip(em["curve"]),
            "p": np.flip(pr["p"]),
            "r": np.flip(pr["r"]),
        }
        numerical_results = {
            "SM": sm,
            "MAE": mae,
            "maxE": em["curve"].max(),
            "avgE": em["curve"].mean(),
            "adpE": em["adp"],
            "maxF": fm["curve"].max(),
            "avgF": fm["curve"].mean(),
            "adpF": fm["adp"],
            "wFm": wfm,
            "HCE": hce,
        }
        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        if not return_ndarray:
            sequential_results = ndarray_to_basetype(sequential_results)
            numerical_results = ndarray_to_basetype(numerical_results)
        return {"sequential": sequential_results, "numerical": numerical_results}


sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
# fmt: off
GRAYSCALE_METRIC_MAPPING = {
    # 灰度数据指标
    "fm": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**sample_gray, beta=0.3)},
    "f1": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**sample_gray, beta=1)},
    "pre": {"handler": py_sod_metrics.PrecisionHandler, "kwargs": sample_gray},
    "rec": {"handler": py_sod_metrics.RecallHandler, "kwargs": sample_gray},
    "iou": {"handler": py_sod_metrics.IOUHandler, "kwargs": sample_gray},
    "dice": {"handler": py_sod_metrics.DICEHandler, "kwargs": sample_gray},
    "spec": {"handler": py_sod_metrics.SpecificityHandler, "kwargs": sample_gray},
    "ber": {"handler": py_sod_metrics.BERHandler, "kwargs": sample_gray},
    "oa": {"handler": py_sod_metrics.OverallAccuracyHandler, "kwargs": sample_gray},
    "kappa": {"handler": py_sod_metrics.KappaHandler, "kwargs": sample_gray},
}
BINARY_METRIC_MAPPING = {
    # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
    "sample_bifm": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**sample_bin, beta=0.3)},
    "sample_bif1": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**sample_bin, beta=1)},
    "sample_bipre": {"handler": py_sod_metrics.PrecisionHandler, "kwargs": sample_bin},
    "sample_birec": {"handler": py_sod_metrics.RecallHandler, "kwargs": sample_bin},
    "sample_biiou": {"handler": py_sod_metrics.IOUHandler, "kwargs": sample_bin},
    "sample_bidice": {"handler": py_sod_metrics.DICEHandler, "kwargs": sample_bin},
    "sample_bispec": {"handler": py_sod_metrics.SpecificityHandler, "kwargs": sample_bin},
    "sample_biber": {"handler": py_sod_metrics.BERHandler, "kwargs": sample_bin},
    "sample_bioa": {"handler": py_sod_metrics.OverallAccuracyHandler, "kwargs": sample_bin},
    "sample_bikappa": {"handler": py_sod_metrics.KappaHandler, "kwargs": sample_bin},
    # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
    "overall_bifm": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**overall_bin, beta=0.3)},
    "overall_bif1": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**overall_bin, beta=1)},
    "overall_bipre": {"handler": py_sod_metrics.PrecisionHandler, "kwargs": overall_bin},
    "overall_birec": {"handler": py_sod_metrics.RecallHandler, "kwargs": overall_bin},
    "overall_biiou": {"handler": py_sod_metrics.IOUHandler, "kwargs": overall_bin},
    "overall_bidice": {"handler": py_sod_metrics.DICEHandler, "kwargs": overall_bin},
    "overall_bispec": {"handler": py_sod_metrics.SpecificityHandler, "kwargs": overall_bin},
    "overall_biber": {"handler": py_sod_metrics.BERHandler, "kwargs": overall_bin},
    "overall_bioa": {"handler": py_sod_metrics.OverallAccuracyHandler, "kwargs": overall_bin},
    "overall_bikappa": {"handler": py_sod_metrics.KappaHandler, "kwargs": overall_bin},
}
SIZEINVARIANCE_METRIC_MAPPING = {
    "handler":{
        "si_fm": {"handler": py_sod_metrics.FmeasureHandler, "kwargs": dict(**sample_gray, beta=0.3)},
        "si_pre": {"handler": py_sod_metrics.PrecisionHandler, "kwargs": dict(with_adaptive=False, with_dynamic=True, sample_based=True)},
        "si_rec": {"handler": py_sod_metrics.RecallHandler, "kwargs": dict(with_adaptive=False, with_dynamic=True, sample_based=True)},
        "si_tpr": {"handler": py_sod_metrics.TPRHandler, "kwargs": dict(with_adaptive=False, with_dynamic=True, sample_based=True)},
        "si_fpr": {"handler": py_sod_metrics.FPRHandler, "kwargs": dict(with_adaptive=False, with_dynamic=True, sample_based=True)},
    },
    "si_fmeasurev2": py_sod_metrics.SizeInvarianceFmeasureV2,
    "si_mae": py_sod_metrics.SizeInvarianceMAE,
}
# fmt: on


class GrayscaleMetricRecorderV2:
    supported_metrics = ["mae", "em", "sm", "wfm", "hce"] + sorted(GRAYSCALE_METRIC_MAPPING.keys())

    def __init__(self, metric_names=("sm", "wfm", "mae", "fmeasure", "em")):
        """
        用于统计各种指标的类，支持更多的指标，更好的兼容性。
        """
        if not metric_names:
            metric_names = self.supported_metrics
        assert all([m in self.supported_metrics for m in metric_names]), f"Only support: {self.supported_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in INDIVADUAL_METRIC_MAPPING:
                self.metric_objs[metric_name] = INDIVADUAL_METRIC_MAPPING[metric_name]()
            else:  # metric_name in GRAYSCALE_METRIC_MAPPING
                if not has_existed:  # only init once
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                metric_handler = GRAYSCALE_METRIC_MAPPING[metric_name]
                self.metric_objs["fmeasurev2"].add_handler(
                    handler_name=metric_name,
                    metric_handler=metric_handler["handler"](**metric_handler["kwargs"]),
                )

    def step(self, pre: np.ndarray, gt: np.ndarray):
        assert pre.shape == gt.shape, (pre.shape, gt.shape)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype)

        for m_obj in self.metric_objs.values():
            m_obj.step(pre, gt)

    def get_all_results(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        sequential_results = {}
        numerical_results = {}
        for m_name, m_obj in self.metric_objs.items():
            info = m_obj.get_results()
            if m_name == "fmeasurev2":
                for _name, results in info.items():
                    dynamic_results = results.get("dynamic")
                    adaptive_results = results.get("adaptive")
                    if dynamic_results is not None:
                        sequential_results[_name] = np.flip(dynamic_results)
                        numerical_results[f"max{_name}"] = dynamic_results.max()
                        numerical_results[f"avg{_name}"] = dynamic_results.mean()
                    if adaptive_results is not None:
                        numerical_results[f"adp{_name}"] = adaptive_results
            else:
                results = info[m_name]
                if m_name in ("wfm", "sm", "mae", "hce"):
                    numerical_results[m_name] = results
                elif m_name in ("fm", "em"):
                    sequential_results[m_name] = np.flip(results["curve"])
                    numerical_results.update(
                        {
                            f"max{m_name}": results["curve"].max(),
                            f"avg{m_name}": results["curve"].mean(),
                            f"adp{m_name}": results["adp"],
                        }
                    )
                else:
                    raise NotImplementedError(m_name)

        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        if not return_ndarray:
            sequential_results = ndarray_to_basetype(sequential_results)
            numerical_results = ndarray_to_basetype(numerical_results)
        return {"sequential": sequential_results, "numerical": numerical_results}

    def show(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        return self.get_all_results(num_bits=num_bits, return_ndarray=return_ndarray)["numerical"]


class BinaryMetricRecorder:
    supported_metrics = ["mae", "sm", "wfm", "hce"] + sorted(BINARY_METRIC_MAPPING.keys())

    def __init__(self, metric_names=("bif1", "biprecision", "birecall", "biiou")):
        """
        用于统计各种指标的类，主要适用于对单通道灰度图计算二值图像的指标。
        """
        if not metric_names:
            metric_names = self.supported_metrics
        assert all([m in self.supported_metrics for m in metric_names]), f"Only support: {self.supported_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in INDIVADUAL_METRIC_MAPPING:
                self.metric_objs[metric_name] = INDIVADUAL_METRIC_MAPPING[metric_name]()
            else:  # metric_name in BINARY_METRIC_MAPPING
                if not has_existed:  # only init once
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                metric_handler = BINARY_METRIC_MAPPING[metric_name]
                self.metric_objs["fmeasurev2"].add_handler(
                    handler_name=metric_name,
                    metric_handler=metric_handler["handler"](**metric_handler["kwargs"]),
                )

    def step(self, pre: np.ndarray, gt: np.ndarray):
        assert pre.shape == gt.shape, (pre.shape, gt.shape)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype)

        for m_obj in self.metric_objs.values():
            m_obj.step(pre, gt)

    def get_all_results(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        numerical_results = {}
        for m_name, m_obj in self.metric_objs.items():
            info = m_obj.get_results()
            if m_name == "fmeasurev2":
                for _name, results in info.items():
                    binary_results = results.get("binary")
                    if binary_results is not None:
                        numerical_results[_name] = binary_results
            else:
                results = info[m_name]
                if m_name in ("mae", "sm", "wfm", "hce"):
                    numerical_results[m_name] = results
                else:
                    raise NotImplementedError(m_name)

        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        if not return_ndarray:
            numerical_results = ndarray_to_basetype(numerical_results)
        return {"numerical": numerical_results}

    def show(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        return self.get_all_results(num_bits=num_bits, return_ndarray=return_ndarray)["numerical"]


class TargetwiseGrayscaleMetricRecorderV2:
    supported_metrics = ["si_mae"] + sorted(SIZEINVARIANCE_METRIC_MAPPING["handler"].keys())

    def __init__(self, metric_names=("si_mae",)):
        if not metric_names:
            metric_names = self.supported_metrics
        assert all([m in self.supported_metrics for m in metric_names]), f"Only support: {self.supported_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in SIZEINVARIANCE_METRIC_MAPPING["handler"]:
                if not has_existed:  # only init once
                    self.metric_objs["si_fmeasurev2"] = SIZEINVARIANCE_METRIC_MAPPING["si_fmeasurev2"]()
                    has_existed = True
                metric_handler = SIZEINVARIANCE_METRIC_MAPPING["handler"][metric_name]
                self.metric_objs["si_fmeasurev2"].add_handler(
                    handler_name=metric_name,
                    metric_handler=metric_handler["handler"](**metric_handler["kwargs"]),
                )
            else:
                self.metric_objs[metric_name] = SIZEINVARIANCE_METRIC_MAPPING[metric_name]()

    def step(self, pre: np.ndarray, gt: np.ndarray):
        assert pre.shape == gt.shape, (pre.shape, gt.shape)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype)

        for m_obj in self.metric_objs.values():
            m_obj.step(pre, gt)

    def cal_auc(self, y, x):
        sorted_idx = np.argsort(x, axis=-1, kind="stable")
        x = np.take_along_axis(x, sorted_idx, axis=-1)
        y = np.take_along_axis(y, sorted_idx, axis=-1)
        return np.trapz(y, x, axis=-1)

    def get_all_results(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        sequential_results = {}
        numerical_results = {}
        for m_name, m_obj in self.metric_objs.items():
            info = m_obj.get_results()

            if m_name == "si_fmeasurev2":
                # AUC-ROC
                if "si_tpr" in info and "si_fpr" in info:
                    ys = info.pop("si_tpr")["dynamic"]  # >=255,>=254,...>=1,>=0
                    xs = info.pop("si_fpr")["dynamic"]
                    if isinstance(ys, list) and isinstance(xs, list):  # Nx[T'x256]
                        auc_results = []
                        for y, x in zip(ys, xs):
                            # NOTE: before calculate the auc, we need to flip the y and x to corresponding to ascending thresholds
                            # because the dynamic results from our metrics is based on the descending order of thresholds, i.e., >=255,>=254,...>=1,>=0
                            y = np.flip(y, axis=-1)
                            x = np.flip(x, axis=-1)
                            auc_results.append(self.cal_auc(y, x).mean())
                        numerical_results["si_sample_auc_roc"] = np.asarray(auc_results).mean()
                    else:  # 256
                        numerical_results["si_overall_auc_roc"] = self.cal_auc(y=ys, x=xs).mean()

                # AUC-PR
                if "si_pre" in info and "si_rec" in info:
                    ys = info.pop("si_pre")["dynamic"]  # >=255,>=254,...>=1,>=0
                    xs = info.pop("si_rec")["dynamic"]
                    if isinstance(ys, list) and isinstance(xs, list):  # Nx[T'x256]
                        auc_results = []
                        for y, x in zip(ys, xs):
                            y = np.flip(y, axis=-1)
                            x = np.flip(x, axis=-1)
                            auc_results.append(self.cal_auc(y, x).mean())
                        numerical_results["si_sample_auc_pr"] = np.asarray(auc_results).mean()
                    else:  # 256
                        numerical_results["si_overall_auc_pr"] = self.cal_auc(y=ys, x=xs).mean()

                for _name, results in info.items():
                    dynamic_results = results.get("dynamic")
                    if dynamic_results is not None:
                        if isinstance(dynamic_results, list):  # Nx[T'x256]
                            max_results = []
                            avg_results = []
                            seq_results = []
                            for s in dynamic_results:
                                max_results.append(s.max(axis=-1).mean())  # 1
                                avg_results.append(s.mean(axis=-1).mean())  # 1
                                seq_results.append(s.mean(axis=0))  # 256
                            seq_results = np.mean(np.asarray(seq_results), axis=0)
                            numerical_results[f"si_sample_max{_name}"] = np.asarray(max_results).mean()
                            numerical_results[f"si_sample_avg{_name}"] = np.asarray(avg_results).mean()
                        else:  # 256
                            seq_results = dynamic_results
                            numerical_results[f"si_overall_max{_name}"] = dynamic_results.max()
                            numerical_results[f"si_overall_avg{_name}"] = dynamic_results.mean()
                        sequential_results[_name] = np.flip(seq_results)

                    adaptive_results = results.get("adaptive")
                    if adaptive_results is not None:
                        numerical_results[f"si_sample_adp{_name}"] = adaptive_results
            else:
                results = info[m_name]
                if m_name in ("si_mae",):
                    numerical_results[m_name] = results
                else:
                    raise NotImplementedError(m_name)

        if num_bits is not None and isinstance(num_bits, int):
            numerical_results = {k: v.round(num_bits) for k, v in numerical_results.items()}
        if not return_ndarray:
            sequential_results = ndarray_to_basetype(sequential_results)
            numerical_results = ndarray_to_basetype(numerical_results)
        return {"sequential": sequential_results, "numerical": numerical_results}

    def show(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        return self.get_all_results(num_bits=num_bits, return_ndarray=return_ndarray)["numerical"]


if __name__ == "__main__":
    data_root = "./test_data"
    mask_root = os.path.join(data_root, "masks")
    pred_root = os.path.join(data_root, "preds")
    masks = [os.path.join(mask_root, f) for f in sorted(os.listdir(mask_root))]
    preds = [os.path.join(pred_root, f) for f in sorted(os.listdir(pred_root))]

    metrics_v1 = GrayscaleMetricRecorderV2(metric_names=GrayscaleMetricRecorderV2.supported_metrics)
    metrics_v2 = BinaryMetricRecorder(metric_names=BinaryMetricRecorder.supported_metrics)
    metrics_v3 = TargetwiseGrayscaleMetricRecorderV2(
        metric_names=TargetwiseGrayscaleMetricRecorderV2.supported_metrics
    )
    for mask, pred in zip(masks, preds):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)
        if pred.shape != mask.shape:
            pred = cv2.resize(pred, dsize=mask.shape[::-1], interpolation=cv2.INTER_LINEAR)

        metrics_v1.step(pred, mask)
        metrics_v2.step(pred, mask)
        metrics_v3.step(pred, mask)
    print(metrics_v1.show())
    print(metrics_v2.show())
    print(metrics_v3.show())
