# -*- coding: utf-8 -*-
# @Time    : 2021/1/4
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import numpy as np

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
}


class MetricRecorderV1:
    def __init__(self):
        """
        用于统计各种指标的类
        https://github.com/lartpang/Py-SOD-VOS-EvalToolkit/blob/81ce89da6813fdd3e22e3f20e3a09fe1e4a1a87c/utils/recorders/metric_recorder.py

        主要应用于旧版本实现中的五个指标，即mae/fm/sm/em/wfm。推荐使用V2版本。
        """
        self.mae = INDIVADUAL_METRIC_MAPPING["mae"]()
        self.fm = INDIVADUAL_METRIC_MAPPING["fm"]()
        self.sm = INDIVADUAL_METRIC_MAPPING["sm"]()
        self.em = INDIVADUAL_METRIC_MAPPING["em"]()
        self.wfm = INDIVADUAL_METRIC_MAPPING["wfm"]()

    def step(self, pre: np.ndarray, gt: np.ndarray):
        assert pre.shape == gt.shape
        assert pre.dtype == np.uint8
        assert gt.dtype == np.uint8

        self.mae.step(pre, gt)
        self.sm.step(pre, gt)
        self.fm.step(pre, gt)
        self.em.step(pre, gt)
        self.wfm.step(pre, gt)

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
BINARY_CLASSIFICATION_METRIC_MAPPING = {
    # 灰度数据指标
    "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
    "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.1),
    "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
    "rec": py_sod_metrics.RecallHandler(**sample_gray),
    "iou": py_sod_metrics.IOUHandler(**sample_gray),
    "dice": py_sod_metrics.DICEHandler(**sample_gray),
    "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
    "ber": py_sod_metrics.BERHandler(**sample_gray),
    "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
    "kappa": py_sod_metrics.KappaHandler(**sample_gray),
    # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
    "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
    "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
    "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
    "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
    "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
    "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
    "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
    "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
    "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
    "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
    # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
    "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
    "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
    "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
    "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
    "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
    "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
    "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
    "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
    "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
    "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
}


class MetricRecorderV2:
    suppoted_metrics = ["mae", "em", "sm", "wfm"] + sorted(
        [
            k
            for k in BINARY_CLASSIFICATION_METRIC_MAPPING.keys()
            if not k.startswith(("sample_", "overall_"))
        ]
    )

    def __init__(self, metric_names=("sm", "wfm", "mae", "fmeasure", "em")):
        """
        用于统计各种指标的类，支持更多的指标，更好的兼容性。
        """
        if not metric_names:
            metric_names = self.suppoted_metrics
        assert all(
            [m in self.suppoted_metrics for m in metric_names]
        ), f"Only support: {self.suppoted_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in INDIVADUAL_METRIC_MAPPING:
                self.metric_objs[metric_name] = INDIVADUAL_METRIC_MAPPING[metric_name]()
            else:  # metric_name in BINARY_CLASSIFICATION_METRIC_MAPPING
                if not has_existed:  # only init once
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                metric_handler = BINARY_CLASSIFICATION_METRIC_MAPPING[metric_name]
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
                if m_name in ("wfm", "sm", "mae"):
                    numerical_results[m_name] = results
                elif m_name == "em":
                    sequential_results[m_name] = np.flip(results["curve"])
                    numerical_results.update(
                        {
                            "maxem": results["curve"].max(),
                            "avgem": results["curve"].mean(),
                            "adpem": results["adp"],
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
    suppoted_metrics = ["mae", "sm", "wfm"] + sorted(
        [
            k
            for k in BINARY_CLASSIFICATION_METRIC_MAPPING.keys()
            if k.startswith(("sample_", "overall_"))
        ]
    )

    def __init__(self, metric_names=("bif1", "biprecision", "birecall", "biiou")):
        """
        用于统计各种指标的类，主要适用于对单通道灰度图计算二值图像的指标。
        """
        if not metric_names:
            metric_names = self.suppoted_metrics
        assert all(
            [m in self.suppoted_metrics for m in metric_names]
        ), f"Only support: {self.suppoted_metrics}"

        self.metric_objs = {}
        has_existed = False
        for metric_name in metric_names:
            if metric_name in INDIVADUAL_METRIC_MAPPING:
                self.metric_objs[metric_name] = INDIVADUAL_METRIC_MAPPING[metric_name]()
            else:  # metric_name in BINARY_CLASSIFICATION_METRIC_MAPPING
                if not has_existed:  # only init once
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                metric_handler = BINARY_CLASSIFICATION_METRIC_MAPPING[metric_name]
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
                if m_name in ("mae", "sm", "wfm"):
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


if __name__ == "__main__":
    data_loader = ...
    model = ...

    cal_total_seg_metrics = MetricRecorderV2()
    for batch in data_loader:
        seg_preds = model(batch)
        for seg_pred in seg_preds:
            mask_array = ...
            cal_total_seg_metrics.step(seg_pred, mask_array)
    fixed_seg_results = cal_total_seg_metrics.show()
