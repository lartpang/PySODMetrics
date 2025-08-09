import json
import os
import sys
import unittest
from pprint import pprint

import cv2
import numpy as np

sys.path.append("..")
import py_sod_metrics

with open("./version_performance.json", encoding="utf-8", mode="r") as f:
    default_results = json.load(f)


def cal_auc(y: np.ndarray, x: np.ndarray):
    assert y.shape == x.shape, (y.shape, x.shape)
    sorted_idx = np.argsort(x, axis=-1, kind="stable")
    y = np.take_along_axis(y, sorted_idx, axis=-1)
    x = np.take_along_axis(x, sorted_idx, axis=-1)
    return np.trapz(y=y, x=x, axis=-1)


def reduce_dynamic_results_for_max_avg(dynamic_results: list):  # Nx[T'x256] -> Nx[T'] -> N -> 1
    max_results = []
    avg_results = []
    for s in dynamic_results:
        max_results.append(s.max(axis=-1).mean())
        avg_results.append(s.mean(axis=-1).mean())
    return np.asarray(max_results).mean(), np.asarray(avg_results).mean()


def reduce_dynamic_results_for_auc(ys: list, xs: list):  # Nx[T'x256] -> Nx[T'] -> N -> 1
    auc_results = []
    for y, x in zip(ys, xs):
        # NOTE: before calculate the auc, we need to flip the y and x to corresponding to ascending thresholds
        # because the dynamic results from our metrics is based on the descending order of thresholds, i.e., >=255,>=254,...>=1,>=0
        y = np.flip(y, -1)
        x = np.flip(x, -1)
        auc_results.append(cal_auc(y=y, x=x).mean())
    return np.asarray(auc_results).mean()


class CheckMetricTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FM = py_sod_metrics.Fmeasure()
        WFM = py_sod_metrics.WeightedFmeasure()
        SM = py_sod_metrics.Smeasure()
        EM = py_sod_metrics.Emeasure()
        MAE = py_sod_metrics.MAE()
        MSIOU = py_sod_metrics.MSIoU(with_dynamic=True, with_adaptive=True, with_binary=True)

        # fmt: off
        sample_gray = dict(with_adaptive=True, with_dynamic=True)
        sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
        overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
        FMv2 = py_sod_metrics.FmeasureV2(
            metric_handlers={
                # 灰度数据指标
                "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
                "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
                "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
                "rec": py_sod_metrics.RecallHandler(**sample_gray),
                "fpr": py_sod_metrics.FPRHandler(**sample_gray),
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
                "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
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
                "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
                "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
                "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
                "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
                "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
                "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
                "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
            }
        )

        # adaptive is not supported for non-sample-based metrics
        overall_gray = dict(with_adaptive=False, with_dynamic=True, sample_based=False)
        SI_MAE = py_sod_metrics.SizeInvarianceMAE()
        SI_FMv2 = py_sod_metrics.SizeInvarianceFmeasureV2(
            metric_handlers={
                "si_sample_fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
                "si_sample_f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
                "si_sample_pre": py_sod_metrics.PrecisionHandler(**sample_gray),
                "si_sample_rec": py_sod_metrics.RecallHandler(**sample_gray),
                "si_sample_fpr": py_sod_metrics.FPRHandler(**sample_gray),
                "si_sample_iou": py_sod_metrics.IOUHandler(**sample_gray),
                "si_sample_dice": py_sod_metrics.DICEHandler(**sample_gray),
                "si_sample_spec": py_sod_metrics.SpecificityHandler(**sample_gray),
                "si_sample_ber": py_sod_metrics.BERHandler(**sample_gray),
                "si_sample_oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
                "si_sample_kappa": py_sod_metrics.KappaHandler(**sample_gray),
                #
                "si_overall_fm": py_sod_metrics.FmeasureHandler(**overall_gray, beta=0.3),
                "si_overall_f1": py_sod_metrics.FmeasureHandler(**overall_gray, beta=1),
                "si_overall_pre": py_sod_metrics.PrecisionHandler(**overall_gray),
                "si_overall_rec": py_sod_metrics.RecallHandler(**overall_gray),
                "si_overall_fpr": py_sod_metrics.FPRHandler(**overall_gray),
                "si_overall_iou": py_sod_metrics.IOUHandler(**overall_gray),
                "si_overall_dice": py_sod_metrics.DICEHandler(**overall_gray),
                "si_overall_spec": py_sod_metrics.SpecificityHandler(**overall_gray),
                "si_overall_ber": py_sod_metrics.BERHandler(**overall_gray),
                "si_overall_oa": py_sod_metrics.OverallAccuracyHandler(**overall_gray),
                "si_overall_kappa": py_sod_metrics.KappaHandler(**overall_gray),
                # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
                "si_sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
                "si_sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
                "si_sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
                "si_sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
                "si_sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
                "si_sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
                "si_sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
                "si_sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
                "si_sample_biber": py_sod_metrics.BERHandler(**sample_bin),
                "si_sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
                "si_sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
                # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
                "si_overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
                "si_overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
                "si_overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
                "si_overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
                "si_overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
                "si_overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
                "si_overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
                "si_overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
                "si_overall_biber": py_sod_metrics.BERHandler(**overall_bin),
                "si_overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
                "si_overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
            }
        )
        # fmt: on

        data_root = "./test_data"
        mask_root = os.path.join(data_root, "masks")
        pred_root = os.path.join(data_root, "preds")
        mask_name_list = sorted(os.listdir(mask_root))

        for i, mask_name in enumerate(mask_name_list):
            print(f"[{i}] Processing {mask_name}...")
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            MAE.step(pred=pred, gt=mask)
            MSIOU.step(pred=pred, gt=mask)
            FMv2.step(pred=pred, gt=mask)
            SI_MAE.step(pred=pred, gt=mask)
            SI_FMv2.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]
        msiou = MSIOU.get_results()
        fmv2 = FMv2.get_results()
        si_mae = SI_MAE.get_results()["si_mae"]
        si_fmv2 = SI_FMv2.get_results()

        cls.curr_results = {
            "MAE": mae,
            "Smeasure": sm,
            "wFmeasure": wfm,
            # "MSIOU": msiou,
            "adpmsiou": msiou["adaptive"],
            "meanmsiou": msiou["dynamic"].mean(),
            "maxmsiou": msiou["dynamic"].max(),
            "sample_bimsiou": msiou["binary"],
            # E-measure for sod
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            # F-measure for sod
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
            # size-invariant
            "si_mae": si_mae,
        }
        # fmt: off
        base_metrics = ["fm", "f1", "pre", "rec", "fpr", "iou", "dice", "spec", "ber", "oa", "kappa"]
        # fmt: on
        for m_name in base_metrics:
            si_sample_max, si_sample_mean = reduce_dynamic_results_for_max_avg(
                si_fmv2[f"si_sample_{m_name}"]["dynamic"]
            )
            cls.curr_results.update(
                {
                    # general form
                    f"adp{m_name}": fmv2[m_name]["adaptive"],
                    f"mean{m_name}": fmv2[m_name]["dynamic"].mean(),
                    f"max{m_name}": fmv2[m_name]["dynamic"].max(),
                    f"sample_bi{m_name}": fmv2[f"sample_bi{m_name}"]["binary"],
                    f"overall_bi{m_name}": fmv2[f"overall_bi{m_name}"]["binary"],
                    # size-invariant
                    # calculate the mean/maximum based on the average fm sequence across all targets from all samples.
                    f"si_overall_mean{m_name}": si_fmv2[f"si_overall_{m_name}"]["dynamic"].mean(),
                    f"si_overall_max{m_name}": si_fmv2[f"si_overall_{m_name}"]["dynamic"].max(),
                    f"si_overall_bi{m_name}": si_fmv2[f"si_overall_bi{m_name}"]["binary"],
                    # calculate the sample-specific mean/maximum based on the sample-based fm sequence with a shape of `(num_targets, 256)`.
                    f"si_sample_mean{m_name}": si_sample_mean,
                    f"si_sample_max{m_name}": si_sample_max,
                    f"si_sample_adp{m_name}": si_fmv2[f"si_sample_{m_name}"]["adaptive"],
                    f"si_sample_bi{m_name}": si_fmv2[f"si_sample_bi{m_name}"]["binary"],
                }
            )
        pr_pre = fmv2["pre"]["dynamic"]  # 256
        pr_rec = fmv2["rec"]["dynamic"]  # 256
        roc_fpr = fmv2["fpr"]["dynamic"]  # tpr is the same as recall
        cls.curr_results["auc_pr"] = cal_auc(y=np.flip(pr_pre, -1), x=np.flip(pr_rec, -1))
        cls.curr_results["auc_roc"] = cal_auc(y=np.flip(pr_rec, -1), x=np.flip(roc_fpr, -1))

        si_overall_pr_pre = si_fmv2["si_overall_pre"]["dynamic"]  # 256
        si_overall_pr_rec = si_fmv2["si_overall_rec"]["dynamic"]  # 256
        si_overall_roc_fpr = si_fmv2["si_overall_fpr"]["dynamic"]  # 256
        cls.curr_results["si_overall_auc_pr"] = cal_auc(
            y=np.flip(si_overall_pr_pre, -1), x=np.flip(si_overall_pr_rec, -1)
        )
        cls.curr_results["si_overall_auc_roc"] = cal_auc(
            y=np.flip(si_overall_pr_rec, -1), x=np.flip(si_overall_roc_fpr, -1)
        )

        si_sample_pr_pre = si_fmv2["si_sample_pre"]["dynamic"]  # Nx[T'x256]
        si_sample_pr_rec = si_fmv2["si_sample_rec"]["dynamic"]  # Nx[T'x256]
        si_sample_roc_fpr = si_fmv2["si_sample_fpr"]["dynamic"]  # Nx[T'x256]
        cls.curr_results["si_sample_auc_pr"] = reduce_dynamic_results_for_auc(ys=si_sample_pr_pre, xs=si_sample_pr_rec)
        cls.curr_results["si_sample_auc_roc"] = reduce_dynamic_results_for_auc(
            ys=si_sample_pr_rec, xs=si_sample_roc_fpr
        )

        print("Current results:")
        pprint(cls.curr_results)
        cls.default_results = default_results["v1_4_3"]  # 68
        si_variant_results = default_results["v1_5_0"]  # 78+6
        for res in [si_variant_results]:
            if any([k in cls.default_results for k in res.keys()]):
                raise ValueError("Some keys will be overwritten by the SI variant results.")
            cls.default_results.update(res)

    def test_sm(self):
        self.assertEqual(self.curr_results["Smeasure"], self.default_results["Smeasure"])

    def test_wfm(self):
        self.assertEqual(self.curr_results["wFmeasure"], self.default_results["wFmeasure"])

    def test_mae(self):
        self.assertEqual(self.curr_results["MAE"], self.default_results["MAE"])

        self.assertEqual(self.curr_results["si_mae"], self.default_results["si_mae"])

    def test_msiou(self):
        # self.assertEqual(self.curr_results["MSIOU"], self.default_results["MSIOU"])
        self.assertEqual(self.curr_results["adpmsiou"], self.default_results["adpmsiou"])
        self.assertEqual(self.curr_results["meanmsiou"], self.default_results["meanmsiou"])
        self.assertEqual(self.curr_results["maxmsiou"], self.default_results["maxmsiou"])
        self.assertEqual(self.curr_results["sample_bimsiou"], self.default_results["sample_bimsiou"])

    def test_fm(self):
        self.assertEqual(self.curr_results["adpFm"], self.default_results["adpFm"])
        self.assertEqual(self.curr_results["meanFm"], self.default_results["meanFm"])
        self.assertEqual(self.curr_results["maxFm"], self.default_results["maxFm"])

        self.assertEqual(self.curr_results["adpfm"], self.default_results["adpfm"])
        self.assertEqual(self.curr_results["meanfm"], self.default_results["meanfm"])
        self.assertEqual(self.curr_results["maxfm"], self.default_results["maxfm"])

        # 对齐v1版本
        self.assertEqual(self.curr_results["adpFm"], self.default_results["adpfm"])
        self.assertEqual(self.curr_results["meanFm"], self.default_results["meanfm"])
        self.assertEqual(self.curr_results["maxFm"], self.default_results["maxfm"])

        self.assertEqual(self.curr_results["sample_bifm"], self.default_results["sample_bifm"])
        self.assertEqual(self.curr_results["overall_bifm"], self.default_results["overall_bifm"])

        self.assertEqual(self.curr_results["si_sample_adpfm"], self.default_results["si_sample_adpfm"])
        self.assertEqual(self.curr_results["si_sample_meanfm"], self.default_results["si_sample_meanfm"])
        self.assertEqual(self.curr_results["si_sample_maxfm"], self.default_results["si_sample_maxfm"])
        self.assertEqual(self.curr_results["si_sample_bifm"], self.default_results["si_sample_bifm"])
        self.assertEqual(self.curr_results["si_overall_meanfm"], self.default_results["si_overall_meanfm"])
        self.assertEqual(self.curr_results["si_overall_maxfm"], self.default_results["si_overall_maxfm"])
        self.assertEqual(self.curr_results["si_overall_bifm"], self.default_results["si_overall_bifm"])

    def test_em(self):
        self.assertEqual(self.curr_results["adpEm"], self.default_results["adpEm"])
        self.assertEqual(self.curr_results["meanEm"], self.default_results["meanEm"])
        self.assertEqual(self.curr_results["maxEm"], self.default_results["maxEm"])

    def test_f1(self):
        self.assertEqual(self.curr_results["adpf1"], self.default_results["adpf1"])
        self.assertEqual(self.curr_results["meanf1"], self.default_results["meanf1"])
        self.assertEqual(self.curr_results["maxf1"], self.default_results["maxf1"])
        self.assertEqual(self.curr_results["sample_bif1"], self.default_results["sample_bif1"])
        self.assertEqual(self.curr_results["overall_bif1"], self.default_results["overall_bif1"])

        self.assertEqual(self.curr_results["si_sample_adpf1"], self.default_results["si_sample_adpf1"])
        self.assertEqual(self.curr_results["si_sample_meanf1"], self.default_results["si_sample_meanf1"])
        self.assertEqual(self.curr_results["si_sample_maxf1"], self.default_results["si_sample_maxf1"])
        self.assertEqual(self.curr_results["si_sample_bif1"], self.default_results["si_sample_bif1"])
        self.assertEqual(self.curr_results["si_overall_meanf1"], self.default_results["si_overall_meanf1"])
        self.assertEqual(self.curr_results["si_overall_maxf1"], self.default_results["si_overall_maxf1"])
        self.assertEqual(self.curr_results["si_overall_bif1"], self.default_results["si_overall_bif1"])

    def test_pre(self):
        self.assertEqual(self.curr_results["adppre"], self.default_results["adppre"])
        self.assertEqual(self.curr_results["meanpre"], self.default_results["meanpre"])
        self.assertEqual(self.curr_results["maxpre"], self.default_results["maxpre"])
        self.assertEqual(self.curr_results["sample_bipre"], self.default_results["sample_bipre"])
        self.assertEqual(self.curr_results["overall_bipre"], self.default_results["overall_bipre"])

        self.assertEqual(self.curr_results["si_sample_adppre"], self.default_results["si_sample_adppre"])
        self.assertEqual(self.curr_results["si_sample_meanpre"], self.default_results["si_sample_meanpre"])
        self.assertEqual(self.curr_results["si_sample_maxpre"], self.default_results["si_sample_maxpre"])
        self.assertEqual(self.curr_results["si_sample_bipre"], self.default_results["si_sample_bipre"])
        self.assertEqual(self.curr_results["si_overall_meanpre"], self.default_results["si_overall_meanpre"])
        self.assertEqual(self.curr_results["si_overall_maxpre"], self.default_results["si_overall_maxpre"])
        self.assertEqual(self.curr_results["si_overall_bipre"], self.default_results["si_overall_bipre"])

    def test_rec(self):
        self.assertEqual(self.curr_results["adprec"], self.default_results["adprec"])
        self.assertEqual(self.curr_results["meanrec"], self.default_results["meanrec"])
        self.assertEqual(self.curr_results["maxrec"], self.default_results["maxrec"])
        self.assertEqual(self.curr_results["sample_birec"], self.default_results["sample_birec"])
        self.assertEqual(self.curr_results["overall_birec"], self.default_results["overall_birec"])

        self.assertEqual(self.curr_results["si_sample_adprec"], self.default_results["si_sample_adprec"])
        self.assertEqual(self.curr_results["si_sample_meanrec"], self.default_results["si_sample_meanrec"])
        self.assertEqual(self.curr_results["si_sample_maxrec"], self.default_results["si_sample_maxrec"])
        self.assertEqual(self.curr_results["si_sample_birec"], self.default_results["si_sample_birec"])
        self.assertEqual(self.curr_results["si_overall_meanrec"], self.default_results["si_overall_meanrec"])
        self.assertEqual(self.curr_results["si_overall_maxrec"], self.default_results["si_overall_maxrec"])
        self.assertEqual(self.curr_results["si_overall_birec"], self.default_results["si_overall_birec"])

    def test_fpr(self):
        self.assertEqual(self.curr_results["adpfpr"], self.default_results["adpfpr"])
        self.assertEqual(self.curr_results["meanfpr"], self.default_results["meanfpr"])
        self.assertEqual(self.curr_results["maxfpr"], self.default_results["maxfpr"])
        self.assertEqual(self.curr_results["sample_bifpr"], self.default_results["sample_bifpr"])
        self.assertEqual(self.curr_results["overall_bifpr"], self.default_results["overall_bifpr"])

        self.assertEqual(self.curr_results["si_sample_adpfpr"], self.default_results["si_sample_adpfpr"])
        self.assertEqual(self.curr_results["si_sample_meanfpr"], self.default_results["si_sample_meanfpr"])
        self.assertEqual(self.curr_results["si_sample_maxfpr"], self.default_results["si_sample_maxfpr"])
        self.assertEqual(self.curr_results["si_sample_bifpr"], self.default_results["si_sample_bifpr"])
        self.assertEqual(self.curr_results["si_overall_meanfpr"], self.default_results["si_overall_meanfpr"])
        self.assertEqual(self.curr_results["si_overall_maxfpr"], self.default_results["si_overall_maxfpr"])
        self.assertEqual(self.curr_results["si_overall_bifpr"], self.default_results["si_overall_bifpr"])

    def test_iou(self):
        self.assertEqual(self.curr_results["adpiou"], self.default_results["adpiou"])
        self.assertEqual(self.curr_results["meaniou"], self.default_results["meaniou"])
        self.assertEqual(self.curr_results["maxiou"], self.default_results["maxiou"])
        self.assertEqual(self.curr_results["sample_biiou"], self.default_results["sample_biiou"])
        self.assertEqual(self.curr_results["overall_biiou"], self.default_results["overall_biiou"])

        self.assertEqual(self.curr_results["si_sample_adpiou"], self.default_results["si_sample_adpiou"])
        self.assertEqual(self.curr_results["si_sample_meaniou"], self.default_results["si_sample_meaniou"])
        self.assertEqual(self.curr_results["si_sample_maxiou"], self.default_results["si_sample_maxiou"])
        self.assertEqual(self.curr_results["si_sample_biiou"], self.default_results["si_sample_biiou"])
        self.assertEqual(self.curr_results["si_overall_meaniou"], self.default_results["si_overall_meaniou"])
        self.assertEqual(self.curr_results["si_overall_maxiou"], self.default_results["si_overall_maxiou"])
        self.assertEqual(self.curr_results["si_overall_biiou"], self.default_results["si_overall_biiou"])

    def test_dice(self):
        self.assertEqual(self.curr_results["adpdice"], self.default_results["adpdice"])
        self.assertEqual(self.curr_results["meandice"], self.default_results["meandice"])
        self.assertEqual(self.curr_results["maxdice"], self.default_results["maxdice"])
        self.assertEqual(self.curr_results["sample_bidice"], self.default_results["sample_bidice"])
        self.assertEqual(self.curr_results["overall_bidice"], self.default_results["overall_bidice"])

        self.assertEqual(self.curr_results["si_sample_adpdice"], self.default_results["si_sample_adpdice"])
        self.assertEqual(self.curr_results["si_sample_meandice"], self.default_results["si_sample_meandice"])
        self.assertEqual(self.curr_results["si_sample_maxdice"], self.default_results["si_sample_maxdice"])
        self.assertEqual(self.curr_results["si_sample_bidice"], self.default_results["si_sample_bidice"])
        self.assertEqual(self.curr_results["si_overall_meandice"], self.default_results["si_overall_meandice"])
        self.assertEqual(self.curr_results["si_overall_maxdice"], self.default_results["si_overall_maxdice"])
        self.assertEqual(self.curr_results["si_overall_bidice"], self.default_results["si_overall_bidice"])

    def test_spec(self):
        self.assertEqual(self.curr_results["adpspec"], self.default_results["adpspec"])
        self.assertEqual(self.curr_results["meanspec"], self.default_results["meanspec"])
        self.assertEqual(self.curr_results["maxspec"], self.default_results["maxspec"])
        self.assertEqual(self.curr_results["sample_bispec"], self.default_results["sample_bispec"])
        self.assertEqual(self.curr_results["overall_bispec"], self.default_results["overall_bispec"])

        self.assertEqual(self.curr_results["si_sample_adpspec"], self.default_results["si_sample_adpspec"])
        self.assertEqual(self.curr_results["si_sample_meanspec"], self.default_results["si_sample_meanspec"])
        self.assertEqual(self.curr_results["si_sample_maxspec"], self.default_results["si_sample_maxspec"])
        self.assertEqual(self.curr_results["si_sample_bispec"], self.default_results["si_sample_bispec"])
        self.assertEqual(self.curr_results["si_overall_meanspec"], self.default_results["si_overall_meanspec"])
        self.assertEqual(self.curr_results["si_overall_maxspec"], self.default_results["si_overall_maxspec"])
        self.assertEqual(self.curr_results["si_overall_bispec"], self.default_results["si_overall_bispec"])

    def test_ber(self):
        self.assertEqual(self.curr_results["adpber"], self.default_results["adpber"])
        self.assertEqual(self.curr_results["meanber"], self.default_results["meanber"])
        self.assertEqual(self.curr_results["maxber"], self.default_results["maxber"])
        self.assertEqual(self.curr_results["sample_biber"], self.default_results["sample_biber"])
        self.assertEqual(self.curr_results["overall_biber"], self.default_results["overall_biber"])

        self.assertEqual(self.curr_results["si_sample_adpber"], self.default_results["si_sample_adpber"])
        self.assertEqual(self.curr_results["si_sample_meanber"], self.default_results["si_sample_meanber"])
        self.assertEqual(self.curr_results["si_sample_maxber"], self.default_results["si_sample_maxber"])
        self.assertEqual(self.curr_results["si_sample_biber"], self.default_results["si_sample_biber"])
        self.assertEqual(self.curr_results["si_overall_meanber"], self.default_results["si_overall_meanber"])
        self.assertEqual(self.curr_results["si_overall_maxber"], self.default_results["si_overall_maxber"])
        self.assertEqual(self.curr_results["si_overall_biber"], self.default_results["si_overall_biber"])

    def test_oa(self):
        self.assertEqual(self.curr_results["adpoa"], self.default_results["adpoa"])
        self.assertEqual(self.curr_results["meanoa"], self.default_results["meanoa"])
        self.assertEqual(self.curr_results["maxoa"], self.default_results["maxoa"])
        self.assertEqual(self.curr_results["sample_bioa"], self.default_results["sample_bioa"])
        self.assertEqual(self.curr_results["overall_bioa"], self.default_results["overall_bioa"])

        self.assertEqual(self.curr_results["si_sample_adpoa"], self.default_results["si_sample_adpoa"])
        self.assertEqual(self.curr_results["si_sample_meanoa"], self.default_results["si_sample_meanoa"])
        self.assertEqual(self.curr_results["si_sample_maxoa"], self.default_results["si_sample_maxoa"])
        self.assertEqual(self.curr_results["si_sample_bioa"], self.default_results["si_sample_bioa"])
        self.assertEqual(self.curr_results["si_overall_meanoa"], self.default_results["si_overall_meanoa"])
        self.assertEqual(self.curr_results["si_overall_maxoa"], self.default_results["si_overall_maxoa"])
        self.assertEqual(self.curr_results["si_overall_bioa"], self.default_results["si_overall_bioa"])

    def test_kappa(self):
        self.assertEqual(self.curr_results["adpkappa"], self.default_results["adpkappa"])
        self.assertEqual(self.curr_results["meankappa"], self.default_results["meankappa"])
        self.assertEqual(self.curr_results["maxkappa"], self.default_results["maxkappa"])
        self.assertEqual(self.curr_results["sample_bikappa"], self.default_results["sample_bikappa"])
        self.assertEqual(self.curr_results["overall_bikappa"], self.default_results["overall_bikappa"])

        self.assertEqual(self.curr_results["si_sample_adpkappa"], self.default_results["si_sample_adpkappa"])
        self.assertEqual(self.curr_results["si_sample_meankappa"], self.default_results["si_sample_meankappa"])
        self.assertEqual(self.curr_results["si_sample_maxkappa"], self.default_results["si_sample_maxkappa"])
        self.assertEqual(self.curr_results["si_sample_bikappa"], self.default_results["si_sample_bikappa"])
        self.assertEqual(self.curr_results["si_overall_meankappa"], self.default_results["si_overall_meankappa"])
        self.assertEqual(self.curr_results["si_overall_maxkappa"], self.default_results["si_overall_maxkappa"])
        self.assertEqual(self.curr_results["si_overall_bikappa"], self.default_results["si_overall_bikappa"])


if __name__ == "__main__":
    unittest.main()
