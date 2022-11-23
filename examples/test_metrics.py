# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys
from pprint import pprint

import cv2

sys.path.append("..")
import py_sod_metrics

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers=[
        py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True),
        py_sod_metrics.PrecisionHandler(with_adaptive=True, with_dynamic=True),
        py_sod_metrics.RecallHandler(with_adaptive=True, with_dynamic=True),
        py_sod_metrics.SpecificityHandler(with_adaptive=True, with_dynamic=True),
        py_sod_metrics.IOUHandler(with_adaptive=True, with_dynamic=True),
        py_sod_metrics.DICEHandler(with_adaptive=True, with_dynamic=True),
    ]
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
fmeasure = fmv2["fmeasure"]
precision = fmv2["precision"]
recall = fmv2["recall"]
specificity = fmv2["specificity"]
iou = fmv2["iou"]
dice = fmv2["dice"]

results = {
    "Smeasure": sm,
    "wFmeasure": wfm,
    "MAE": mae,
    "adpEm": em["adp"],
    "meanEm": em["curve"].mean(),
    "maxEm": em["curve"].max(),
    "adpFm": fm["adp"],
    "meanFm": fm["curve"].mean(),
    "maxFm": fm["curve"].max(),
    "adpfm2": fmeasure["adaptive"],
    "meanfm2": fmeasure["dynamic"].mean(),
    "maxfm2": fmeasure["dynamic"].max(),
    "adpprev2": precision["adaptive"],
    "meanprev2": precision["dynamic"].mean(),
    "maxprev2": precision["dynamic"].max(),
    "adprecv2": recall["adaptive"],
    "meanrecv2": recall["dynamic"].mean(),
    "maxrecv2": recall["dynamic"].max(),
    "adpspecv2": specificity["adaptive"],
    "meanspecv2": specificity["dynamic"].mean(),
    "maxspecv2": specificity["dynamic"].max(),
    "adpiouv2": iou["adaptive"],
    "meaniouv2": iou["dynamic"].mean(),
    "maxiouv2": iou["dynamic"].max(),
    "adpdicev2": dice["adaptive"],
    "meandicev2": dice["dynamic"].mean(),
    "maxdicev2": dice["dynamic"].max(),
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
        "adpdicev2": 0.5801020564379223,
        "adpfm2": 0.5816750824038355,
        "adpiouv2": 0.5141023436626048,
        "adpprev2": 0.583200007681871,
        "adprecv2": 0.5777548546727481,
        "adpspecv2": 0.9512882075256152,
        "maxEm": 0.9669544828922699,
        "maxFm": 0.5886784581120638,
        "maxdicev2": 0.5830613926289557,
        "maxfm2": 0.5886784581120638,
        "maxiouv2": 0.5201569938888494,
        "maxprev2": 0.6396783912301717,
        "maxrecv2": 0.6666666666666666,
        "maxspecv2": 0.9965927890353435,
        "meanEm": 0.9566258293508704,
        "meanFm": 0.577051059518767,
        "meandicev2": 0.5689913551800527,
        "meanfm2": 0.577051059518767,
        "meaniouv2": 0.49816648786971,
        "meanprev2": 0.5857695537152126,
        "meanrecv2": 0.5599653001125341,
        "meanspecv2": 0.9742186408675534,
        "wFmeasure": 0.5579812753638986,
    },
}

pprint(results)
pprint({k: default_value - results[k] for k, default_value in default_results["v1_4_0"].items()})
