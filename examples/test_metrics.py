# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys
import unittest
import cv2

sys.path.append("..")
import py_sod_metrics

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()

sample_binary = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_binary = dict(
    with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False
)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        # 灰度数据指标
        "fm": py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True, beta=0.1),
        "pre": py_sod_metrics.PrecisionHandler(with_adaptive=True, with_dynamic=True),
        "rec": py_sod_metrics.RecallHandler(with_adaptive=True, with_dynamic=True),
        "iou": py_sod_metrics.IOUHandler(with_adaptive=True, with_dynamic=True),
        "dice": py_sod_metrics.DICEHandler(with_adaptive=True, with_dynamic=True),
        "spec": py_sod_metrics.SpecificityHandler(with_adaptive=True, with_dynamic=True),
        "ber": py_sod_metrics.BERHandler(with_adaptive=True, with_dynamic=True),
        # 二值化数据指标的特殊情况一：各个样本独立计算指标后取平均
        "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_binary, beta=0.3),
        "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_binary, beta=1),
        "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_binary),
        "sample_birec": py_sod_metrics.RecallHandler(**sample_binary),
        "sample_biiou": py_sod_metrics.IOUHandler(**sample_binary),
        "sample_bidice": py_sod_metrics.DICEHandler(**sample_binary),
        "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_binary),
        "sample_biber": py_sod_metrics.BERHandler(**sample_binary),
        # 二值化数据指标的特殊情况二：汇总所有样本的tp、fp、tn、fn后整体计算指标
        "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_binary, beta=0.3),
        "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_binary, beta=1),
        "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_binary),
        "overall_birec": py_sod_metrics.RecallHandler(**overall_binary),
        "overall_biiou": py_sod_metrics.IOUHandler(**overall_binary),
        "overall_bidice": py_sod_metrics.DICEHandler(**overall_binary),
        "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_binary),
        "overall_biber": py_sod_metrics.BERHandler(**overall_binary),
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
        "meanpre": 0.5857695537152126,
        "meanrec": 0.5599653001125341,
        "meanspec": 0.9742186408675534,
        "overall_biber": 0.08527759498137788,
        "overall_bidice": 0.8510675335753018,
        "overall_bif1": 0.8510675335753017,
        "overall_bifm": 0.8525259082995088,
        "overall_biiou": 0.740746352327995,
        "overall_bipre": 0.8537799277020065,
        "overall_birec": 0.8483723190115916,
        "overall_bispec": 0.9810724910256526,
        "sample_biber": 0.23037858807333392,
        "sample_bidice": 0.5738376903441331,
        "sample_bif1": 0.5738376903441331,
        "sample_bifm": 0.5829998670906196,
        "sample_biiou": 0.5039622042094377,
        "sample_bipre": 0.5916996553523113,
        "sample_birec": 0.5592859147614985,
        "sample_bispec": 0.9799569090918337,
        "wFmeasure": 0.5579812753638986,
    },
}

class CheckMetricTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = default_results["v1_4_0"]

    def test_sm(self):
        self.assertEqual(sm, self.results["Smeasure"])

    def test_wfm(self):
        self.assertEqual(wfm, self.results["wFmeasure"])

    def test_mae(self):
        self.assertEqual(mae, self.results["MAE"])

    def test_fm(self):
        self.assertEqual(fm["adp"], self.results["adpFm"])
        self.assertEqual(fm["curve"].mean(), self.results["meanFm"])
        self.assertEqual(fm["curve"].max(), self.results["maxFm"])

    def test_em(self):
        self.assertEqual(em["adp"], self.results["adpEm"])
        self.assertEqual(em["curve"].mean(), self.results["meanEm"])
        self.assertEqual(em["curve"].max(), self.results["maxEm"])

    def test_fmv2(self):
        self.assertEqual(fmv2["fm"]["adaptive"], self.results["adpfm"])
        self.assertEqual(fmv2["fm"]["dynamic"].mean(), self.results["meanfm"])
        self.assertEqual(fmv2["fm"]["dynamic"].max(), self.results["maxfm"])
        # 对齐v1版本
        self.assertEqual(fmv2["fm"]["adaptive"], self.results["adpFm"])
        self.assertEqual(fmv2["fm"]["dynamic"].mean(), self.results["meanFm"])
        self.assertEqual(fmv2["fm"]["dynamic"].max(), self.results["maxFm"])

        self.assertEqual(fmv2["f1"]["adaptive"], self.results["adpf1"])
        self.assertEqual(fmv2["f1"]["dynamic"].mean(), self.results["meanf1"])
        self.assertEqual(fmv2["f1"]["dynamic"].max(), self.results["maxf1"])

        self.assertEqual(fmv2["pre"]["adaptive"], self.results["adppre"])
        self.assertEqual(fmv2["pre"]["dynamic"].mean(), self.results["meanpre"])
        self.assertEqual(fmv2["pre"]["dynamic"].max(), self.results["maxpre"])

        self.assertEqual(fmv2["rec"]["adaptive"], self.results["adprec"])
        self.assertEqual(fmv2["rec"]["dynamic"].mean(), self.results["meanrec"])
        self.assertEqual(fmv2["rec"]["dynamic"].max(), self.results["maxrec"])

        self.assertEqual(fmv2["spec"]["adaptive"], self.results["adpspec"])
        self.assertEqual(fmv2["spec"]["dynamic"].mean(), self.results["meanspec"])
        self.assertEqual(fmv2["spec"]["dynamic"].max(), self.results["maxspec"])

        self.assertEqual(fmv2["iou"]["adaptive"], self.results["adpiou"])
        self.assertEqual(fmv2["iou"]["dynamic"].mean(), self.results["meaniou"])
        self.assertEqual(fmv2["iou"]["dynamic"].max(), self.results["maxiou"])

        self.assertEqual(fmv2["dice"]["adaptive"], self.results["adpdice"])
        self.assertEqual(fmv2["dice"]["dynamic"].mean(), self.results["meandice"])
        self.assertEqual(fmv2["dice"]["dynamic"].max(), self.results["maxdice"])

        self.assertEqual(fmv2["ber"]["adaptive"], self.results["adpber"])
        self.assertEqual(fmv2["ber"]["dynamic"].mean(), self.results["meanber"])
        self.assertEqual(fmv2["ber"]["dynamic"].max(), self.results["maxber"])

        self.assertEqual(fmv2["sample_bifm"]["binary"], self.results["sample_bifm"])
        self.assertEqual(fmv2["sample_bif1"]["binary"], self.results["sample_bif1"])
        self.assertEqual(fmv2["sample_bipre"]["binary"], self.results["sample_bipre"])
        self.assertEqual(fmv2["sample_birec"]["binary"], self.results["sample_birec"])
        self.assertEqual(fmv2["sample_biiou"]["binary"], self.results["sample_biiou"])
        self.assertEqual(fmv2["sample_bidice"]["binary"], self.results["sample_bidice"])
        self.assertEqual(fmv2["sample_bispec"]["binary"], self.results["sample_bispec"])
        self.assertEqual(fmv2["sample_biber"]["binary"], self.results["sample_biber"])

        self.assertEqual(fmv2["overall_bifm"]["binary"], self.results["overall_bifm"])
        self.assertEqual(fmv2["overall_bif1"]["binary"], self.results["overall_bif1"])
        self.assertEqual(fmv2["overall_bipre"]["binary"], self.results["overall_bipre"])
        self.assertEqual(fmv2["overall_birec"]["binary"], self.results["overall_birec"])
        self.assertEqual(fmv2["overall_biiou"]["binary"], self.results["overall_biiou"])
        self.assertEqual(fmv2["overall_bidice"]["binary"], self.results["overall_bidice"])
        self.assertEqual(fmv2["overall_bispec"]["binary"], self.results["overall_bispec"])
        self.assertEqual(fmv2["overall_biber"]["binary"], self.results["overall_biber"])


if __name__ == "__main__":
    unittest.main()
