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
BINARY_CLASSIFICATION_METRIC_MAPPING = {
    "fmeasure": py_sod_metrics.FmeasureHandler(with_dynamic=True, with_adaptive=True, beta=0.3),
    "precision": py_sod_metrics.PrecisionHandler(with_dynamic=False, with_adaptive=False),  # close
    "recall": py_sod_metrics.RecallHandler(with_dynamic=False, with_adaptive=False),  # close
    "iou": py_sod_metrics.IOUHandler(with_dynamic=True, with_adaptive=True),
    "dice": py_sod_metrics.DICEHandler(with_dynamic=True, with_adaptive=True),
    "specificity": py_sod_metrics.SpecificityHandler(with_dynamic=True, with_adaptive=True),
}


class CalTotalMetricV1:
    def __init__(self):
        """
        用于统计各种指标的类
        https://github.com/lartpang/Py-SOD-VOS-EvalToolkit/blob/81ce89da6813fdd3e22e3f20e3a09fe1e4a1a87c/utils/recorders/metric_recorder.py
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


class CalTotalMetricV2:
    # 'fm' is replaced by 'fmeasure' in BINARY_CLASSIFICATION_METRIC_MAPPING
    suppoted_metrics = ["mae", "em", "sm", "wfm"] + sorted(
        BINARY_CLASSIFICATION_METRIC_MAPPING.keys()
    )

    def __init__(self, metric_names=None):
        """
        用于统计各种指标的类
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
                if not has_existed:
                    self.metric_objs["fmeasurev2"] = py_sod_metrics.FmeasureV2()
                    has_existed = True
                handler = BINARY_CLASSIFICATION_METRIC_MAPPING[metric_name]
                self.metric_objs["fmeasurev2"].add_handler(handler)

    def update(self, pre: np.ndarray, gt: np.ndarray):
        assert pre.shape == gt.shape, (pre.shape, gt.shape)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype)

        for m_obj in self.metric_objs.values():
            m_obj.step(pre, gt)

    def show(self, num_bits: int = 3, return_ndarray: bool = False) -> dict:
        """
        返回指标计算结果：

        - 曲线数据(sequential)
        - 数值指标(numerical)
        """
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
                            "maxe": results["curve"].max(),
                            "avge": results["curve"].mean(),
                            "adpe": results["adp"],
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


if __name__ == "__main__":
    data_loader = ...
    model = ...

    cal_total_seg_metrics = CalTotalMetric()
    for batch in data_loader:
        seg_preds = model(batch)
        for seg_pred in seg_preds:
            mask_array = ...
            cal_total_seg_metrics.step(seg_pred, mask_array)
    fixed_seg_results = cal_total_seg_metrics.get_results()
