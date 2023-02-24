# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys
import unittest
from pprint import pprint

import cv2

sys.path.append("..")
import py_sod_metrics

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()

sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
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
)

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
    FMv2.step(pred=pred, gt=mask)

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
fmv2 = FMv2.get_results()

curr_results = {
    "MAE": mae,
    "Smeasure": sm,
    "wFmeasure": wfm,
    # E-measure for sod
    "adpEm": em["adp"],
    "meanEm": em["curve"].mean(),
    "maxEm": em["curve"].max(),
    # F-measure for sod
    "adpFm": fm["adp"],
    "meanFm": fm["curve"].mean(),
    "maxFm": fm["curve"].max(),
    # general F-measure
    "adpfm": fmv2["fm"]["adaptive"],
    "meanfm": fmv2["fm"]["dynamic"].mean(),
    "maxfm": fmv2["fm"]["dynamic"].max(),
    "sample_bifm": fmv2["sample_bifm"]["binary"],
    "overall_bifm": fmv2["overall_bifm"]["binary"],
    # precision
    "adppre": fmv2["pre"]["adaptive"],
    "meanpre": fmv2["pre"]["dynamic"].mean(),
    "maxpre": fmv2["pre"]["dynamic"].max(),
    "sample_bipre": fmv2["sample_bipre"]["binary"],
    "overall_bipre": fmv2["overall_bipre"]["binary"],
    # recall
    "adprec": fmv2["rec"]["adaptive"],
    "meanrec": fmv2["rec"]["dynamic"].mean(),
    "maxrec": fmv2["rec"]["dynamic"].max(),
    "sample_birec": fmv2["sample_birec"]["binary"],
    "overall_birec": fmv2["overall_birec"]["binary"],
    # dice
    "adpdice": fmv2["dice"]["adaptive"],
    "meandice": fmv2["dice"]["dynamic"].mean(),
    "maxdice": fmv2["dice"]["dynamic"].max(),
    "sample_bidice": fmv2["sample_bidice"]["binary"],
    "overall_bidice": fmv2["overall_bidice"]["binary"],
    # iou
    "adpiou": fmv2["iou"]["adaptive"],
    "meaniou": fmv2["iou"]["dynamic"].mean(),
    "maxiou": fmv2["iou"]["dynamic"].max(),
    "sample_biiou": fmv2["sample_biiou"]["binary"],
    "overall_biiou": fmv2["overall_biiou"]["binary"],
    # f1 score
    "adpf1": fmv2["f1"]["adaptive"],
    "meanf1": fmv2["f1"]["dynamic"].mean(),
    "maxf1": fmv2["f1"]["dynamic"].max(),
    "sample_bif1": fmv2["sample_bif1"]["binary"],
    "overall_bif1": fmv2["overall_bif1"]["binary"],
    # specificity
    "adpspec": fmv2["spec"]["adaptive"],
    "meanspec": fmv2["spec"]["dynamic"].mean(),
    "maxspec": fmv2["spec"]["dynamic"].max(),
    "sample_bispec": fmv2["sample_bispec"]["binary"],
    "overall_bispec": fmv2["overall_bispec"]["binary"],
    # ber
    "adpber": fmv2["ber"]["adaptive"],
    "meanber": fmv2["ber"]["dynamic"].mean(),
    "maxber": fmv2["ber"]["dynamic"].max(),
    "sample_biber": fmv2["sample_biber"]["binary"],
    "overall_biber": fmv2["overall_biber"]["binary"],
    # overall accuracy
    "adpoa": fmv2["oa"]["adaptive"],
    "meanoa": fmv2["oa"]["dynamic"].mean(),
    "maxoa": fmv2["oa"]["dynamic"].max(),
    "sample_bioa": fmv2["sample_bioa"]["binary"],
    "overall_bioa": fmv2["overall_bioa"]["binary"],
    # kappa
    "adpkappa": fmv2["kappa"]["adaptive"],
    "meankappa": fmv2["kappa"]["dynamic"].mean(),
    "maxkappa": fmv2["kappa"]["dynamic"].max(),
    "sample_bikappa": fmv2["sample_bikappa"]["binary"],
    "overall_bikappa": fmv2["overall_bikappa"]["binary"],
}

default_results = {
    "v1_2_3": {
        "Smeasure": 0.9029763868504661,
        "wFmeasure": 0.5579812753638986,
        "MAE": 0.03705558476661653,
        "adpEm": 0.9408760066970631,
        "meanEm": 0.9566258293508715,
        "maxEm": 0.966954482892271,
        "adpFm": 0.5816750824038355,
        "meanFm": 0.577051059518767,
        "maxFm": 0.5886784581120638,
    },
    "v1_3_0": {
        "Smeasure": 0.9029761578759272,
        "wFmeasure": 0.5579812753638986,
        "MAE": 0.03705558476661653,
        "adpEm": 0.9408760066970617,
        "meanEm": 0.9566258293508704,
        "maxEm": 0.9669544828922699,
        "adpFm": 0.5816750824038355,
        "meanFm": 0.577051059518767,
        "maxFm": 0.5886784581120638,
    },
    "v1_4_0": {
        "MAE": 0.03705558476661653,
        "Smeasure": 0.9029761578759272,
        "adpEm": 0.9408760066970617,
        "adpFm": 0.5816750824038355,
        "adpber": 0.2354784689008184,
        "adpdice": 0.5801020564379223,
        "adpf1": 0.5825795996723205,
        "adpfm": 0.5816750824038355,
        "adpiou": 0.5141023436626048,
        "adpkappa": 0.6568702977598276,
        "adpoa": 0.9391947016812359,
        "adppre": 0.583200007681871,
        "adprec": 0.5777548546727481,
        "adpspec": 0.9512882075256152,
        "maxEm": 0.9669544828922699,
        "maxFm": 0.5886784581120638,
        "maxber": 0.6666666666666666,
        "maxdice": 0.5830613926289557,
        "maxf1": 0.6031100666167747,
        "maxfm": 0.5886784581120638,
        "maxiou": 0.5201569938888494,
        "maxkappa": 0.6759493461328753,
        "maxoa": 0.9654783867686053,
        "maxpre": 0.6396783912301717,
        "maxrec": 0.6666666666666666,
        "maxspec": 0.9965927890353435,
        "meanEm": 0.9566258293508704,
        "meanFm": 0.577051059518767,
        "meanber": 0.23290802950995626,
        "meandice": 0.5689913551800527,
        "meanf1": 0.5821115124232528,
        "meanfm": 0.577051059518767,
        "meaniou": 0.49816648786971,
        "meankappa": 0.6443053495487194,
        "meanoa": 0.9596413706286032,
        "meanpre": 0.5857695537152126,
        "meanrec": 0.5599653001125341,
        "meanspec": 0.9742186408675534,
        "overall_biber": 0.08527759498137788,
        "overall_bidice": 0.8510675335753018,
        "overall_bif1": 0.8510675335753017,
        "overall_bifm": 0.8525259082995088,
        "overall_biiou": 0.740746352327995,
        "overall_bikappa": 0.7400114676102276,
        "overall_bioa": 0.965778,
        "overall_bipre": 0.8537799277020065,
        "overall_birec": 0.8483723190115916,
        "overall_bispec": 0.9810724910256526,
        "sample_biber": 0.23037858807333392,
        "sample_bidice": 0.5738376903441331,
        "sample_bif1": 0.5738376903441331,
        "sample_bifm": 0.5829998670906196,
        "sample_biiou": 0.5039622042094377,
        "sample_bikappa": 0.6510635726572914,
        "sample_bioa": 0.964811758770181,
        "sample_bipre": 0.5916996553523113,
        "sample_birec": 0.5592859147614985,
        "sample_bispec": 0.9799569090918337,
        "wFmeasure": 0.5579812753638986,
    },
}


class CheckMetricTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Current results:")
        pprint(curr_results)
        cls.default_results = default_results["v1_4_0"]

    def test_sm(self):
        self.assertEqual(curr_results["Smeasure"], self.default_results["Smeasure"])

    def test_wfm(self):
        self.assertEqual(curr_results["wFmeasure"], self.default_results["wFmeasure"])

    def test_mae(self):
        self.assertEqual(curr_results["MAE"], self.default_results["MAE"])

    def test_fm(self):
        self.assertEqual(curr_results["adpFm"], self.default_results["adpFm"])
        self.assertEqual(curr_results["meanFm"], self.default_results["meanFm"])
        self.assertEqual(curr_results["maxFm"], self.default_results["maxFm"])

        self.assertEqual(curr_results["adpfm"], self.default_results["adpfm"])
        self.assertEqual(curr_results["meanfm"], self.default_results["meanfm"])
        self.assertEqual(curr_results["maxfm"], self.default_results["maxfm"])

        # 对齐v1版本
        self.assertEqual(curr_results["adpFm"], self.default_results["adpfm"])
        self.assertEqual(curr_results["meanFm"], self.default_results["meanfm"])
        self.assertEqual(curr_results["maxFm"], self.default_results["maxfm"])

        self.assertEqual(curr_results["sample_bifm"], self.default_results["sample_bifm"])
        self.assertEqual(curr_results["overall_bifm"], self.default_results["overall_bifm"])

    def test_em(self):
        self.assertEqual(curr_results["adpEm"], self.default_results["adpEm"])
        self.assertEqual(curr_results["meanEm"], self.default_results["meanEm"])
        self.assertEqual(curr_results["maxEm"], self.default_results["maxEm"])

    def test_f1(self):
        self.assertEqual(curr_results["adpf1"], self.default_results["adpf1"])
        self.assertEqual(curr_results["meanf1"], self.default_results["meanf1"])
        self.assertEqual(curr_results["maxf1"], self.default_results["maxf1"])
        self.assertEqual(curr_results["sample_bif1"], self.default_results["sample_bif1"])
        self.assertEqual(curr_results["overall_bif1"], self.default_results["overall_bif1"])

    def test_pre(self):
        self.assertEqual(curr_results["adppre"], self.default_results["adppre"])
        self.assertEqual(curr_results["meanpre"], self.default_results["meanpre"])
        self.assertEqual(curr_results["maxpre"], self.default_results["maxpre"])
        self.assertEqual(curr_results["sample_bipre"], self.default_results["sample_bipre"])
        self.assertEqual(curr_results["overall_bipre"], self.default_results["overall_bipre"])

    def test_rec(self):
        self.assertEqual(curr_results["adprec"], self.default_results["adprec"])
        self.assertEqual(curr_results["meanrec"], self.default_results["meanrec"])
        self.assertEqual(curr_results["maxrec"], self.default_results["maxrec"])
        self.assertEqual(curr_results["sample_birec"], self.default_results["sample_birec"])
        self.assertEqual(curr_results["overall_birec"], self.default_results["overall_birec"])

    def test_iou(self):
        self.assertEqual(curr_results["adpiou"], self.default_results["adpiou"])
        self.assertEqual(curr_results["meaniou"], self.default_results["meaniou"])
        self.assertEqual(curr_results["maxiou"], self.default_results["maxiou"])
        self.assertEqual(curr_results["sample_biiou"], self.default_results["sample_biiou"])
        self.assertEqual(curr_results["overall_biiou"], self.default_results["overall_biiou"])

    def test_dice(self):
        self.assertEqual(curr_results["adpdice"], self.default_results["adpdice"])
        self.assertEqual(curr_results["meandice"], self.default_results["meandice"])
        self.assertEqual(curr_results["maxdice"], self.default_results["maxdice"])
        self.assertEqual(curr_results["sample_bidice"], self.default_results["sample_bidice"])
        self.assertEqual(curr_results["overall_bidice"], self.default_results["overall_bidice"])

    def test_spec(self):
        self.assertEqual(curr_results["adpspec"], self.default_results["adpspec"])
        self.assertEqual(curr_results["meanspec"], self.default_results["meanspec"])
        self.assertEqual(curr_results["maxspec"], self.default_results["maxspec"])
        self.assertEqual(curr_results["sample_bispec"], self.default_results["sample_bispec"])
        self.assertEqual(curr_results["overall_bispec"], self.default_results["overall_bispec"])

    def test_ber(self):
        self.assertEqual(curr_results["adpber"], self.default_results["adpber"])
        self.assertEqual(curr_results["meanber"], self.default_results["meanber"])
        self.assertEqual(curr_results["maxber"], self.default_results["maxber"])
        self.assertEqual(curr_results["sample_biber"], self.default_results["sample_biber"])
        self.assertEqual(curr_results["overall_biber"], self.default_results["overall_biber"])

    def test_oa(self):
        self.assertEqual(curr_results["adpoa"], self.default_results["adpoa"])
        self.assertEqual(curr_results["meanoa"], self.default_results["meanoa"])
        self.assertEqual(curr_results["maxoa"], self.default_results["maxoa"])
        self.assertEqual(curr_results["sample_bioa"], self.default_results["sample_bioa"])
        self.assertEqual(curr_results["overall_bioa"], self.default_results["overall_bioa"])

    def test_kappa(self):
        self.assertEqual(curr_results["adpkappa"], self.default_results["adpkappa"])
        self.assertEqual(curr_results["meankappa"], self.default_results["meankappa"])
        self.assertEqual(curr_results["maxkappa"], self.default_results["maxkappa"])
        self.assertEqual(curr_results["sample_bikappa"], self.default_results["sample_bikappa"])
        self.assertEqual(curr_results["overall_bikappa"], self.default_results["overall_bikappa"])


if __name__ == "__main__":
    unittest.main()
