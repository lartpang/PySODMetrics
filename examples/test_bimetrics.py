import copy
import logging
import os
import sys

import cv2
import numpy as np

sys.path.append("..")
import py_sod_metrics

logging.basicConfig(level=logging.DEBUG)


def compare_unnormalized(pred_files, mask_files):
    overall_bin = dict(
        with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False
    )
    # single iteration
    sample_recorder = py_sod_metrics.FmeasureV2(
        metric_handlers={
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
    whole_recorder = copy.deepcopy(sample_recorder)

    base_h = base_w = 256

    preds = []
    masks = []
    for pred_path, mask_path in zip(pred_files, mask_files):
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        assert pred is not None, pred_path
        pred = cv2.resize(pred, dsize=(base_w, base_h), interpolation=cv2.INTER_LINEAR)
        preds.append(pred)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, mask_path
        mask = cv2.resize(mask, dsize=(base_w, base_h), interpolation=cv2.INTER_LINEAR)
        masks.append(mask)

        pred = pred / 255
        mask = mask > 127
        sample_recorder.step(pred=pred, gt=mask, normalize=False)
    sample_results = sample_recorder.get_results()
    sample_info = {
        "overall_bifm": sample_results["overall_bifm"]["binary"],
        "overall_bipre": sample_results["overall_bipre"]["binary"],
        "overall_birec": sample_results["overall_birec"]["binary"],
        "overall_bifpr": sample_results["overall_bifpr"]["binary"],
        "overall_bidice": sample_results["overall_bidice"]["binary"],
        "overall_biiou": sample_results["overall_biiou"]["binary"],
        "overall_bif1": sample_results["overall_bif1"]["binary"],
        "overall_bispec": sample_results["overall_bispec"]["binary"],
        "overall_biber": sample_results["overall_biber"]["binary"],
        "overall_bioa": sample_results["overall_bioa"]["binary"],
        "overall_bikappa": sample_results["overall_bikappa"]["binary"],
    }

    preds = np.concatenate(preds, axis=-1)  # H,n*W
    masks = np.concatenate(masks, axis=-1)
    preds = preds / 255
    masks = masks > 127
    whole_recorder.step(pred=preds, gt=masks, normalize=False)
    whole_results = whole_recorder.get_results()
    whole_info = {
        "overall_bifm": whole_results["overall_bifm"]["binary"],
        "overall_bipre": whole_results["overall_bipre"]["binary"],
        "overall_birec": whole_results["overall_birec"]["binary"],
        "overall_bifpr": whole_results["overall_bifpr"]["binary"],
        "overall_bidice": whole_results["overall_bidice"]["binary"],
        "overall_biiou": whole_results["overall_biiou"]["binary"],
        "overall_bif1": whole_results["overall_bif1"]["binary"],
        "overall_bispec": whole_results["overall_bispec"]["binary"],
        "overall_biber": whole_results["overall_biber"]["binary"],
        "overall_bioa": whole_results["overall_bioa"]["binary"],
        "overall_bikappa": whole_results["overall_bikappa"]["binary"],
    }

    for name, sample_value in sample_info.items():
        whole_value = whole_info[name]
        # 此时的结果应该是一致的
        if sample_value == whole_value:
            logging.info(f"[normalized] {name} passed!")
        else:
            logging.warning(
                f"[normalized] {name} should be equal: {sample_value} vs {whole_value}"
            )


def compare_normalized(pred_files, mask_files):
    overall_bin = dict(
        with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False
    )
    # single iteration
    sample_recorder = py_sod_metrics.FmeasureV2(
        metric_handlers={
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
    whole_recorder = copy.deepcopy(sample_recorder)

    base_h = base_w = 256

    preds = []
    masks = []
    for pred_path, mask_path in zip(pred_files, mask_files):
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        assert pred is not None, pred_path
        pred = cv2.resize(pred, dsize=(base_w, base_h), interpolation=cv2.INTER_LINEAR)
        preds.append(pred)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, mask_path
        mask = cv2.resize(mask, dsize=(base_w, base_h), interpolation=cv2.INTER_LINEAR)
        masks.append(mask)

        sample_recorder.step(pred=pred, gt=mask, normalize=True)
    sample_results = sample_recorder.get_results()
    sample_info = {
        "overall_bifm": sample_results["overall_bifm"]["binary"],
        "overall_bipre": sample_results["overall_bipre"]["binary"],
        "overall_birec": sample_results["overall_birec"]["binary"],
        "overall_bifpr": sample_results["overall_bifpr"]["binary"],
        "overall_bidice": sample_results["overall_bidice"]["binary"],
        "overall_biiou": sample_results["overall_biiou"]["binary"],
        "overall_bif1": sample_results["overall_bif1"]["binary"],
        "overall_bispec": sample_results["overall_bispec"]["binary"],
        "overall_biber": sample_results["overall_biber"]["binary"],
        "overall_bioa": sample_results["overall_bioa"]["binary"],
        "overall_bikappa": sample_results["overall_bikappa"]["binary"],
    }

    preds = np.concatenate(preds, axis=-1)  # H,n*W
    masks = np.concatenate(masks, axis=-1)
    whole_recorder.step(pred=preds, gt=masks, normalize=True)
    whole_results = whole_recorder.get_results()
    whole_info = {
        "overall_bifm": whole_results["overall_bifm"]["binary"],
        "overall_bipre": whole_results["overall_bipre"]["binary"],
        "overall_birec": whole_results["overall_birec"]["binary"],
        "overall_bifpr": whole_results["overall_bifpr"]["binary"],
        "overall_bidice": whole_results["overall_bidice"]["binary"],
        "overall_biiou": whole_results["overall_biiou"]["binary"],
        "overall_bif1": whole_results["overall_bif1"]["binary"],
        "overall_bispec": whole_results["overall_bispec"]["binary"],
        "overall_biber": whole_results["overall_biber"]["binary"],
        "overall_bioa": whole_results["overall_bioa"]["binary"],
        "overall_bikappa": whole_results["overall_bikappa"]["binary"],
    }

    for name, sample_value in sample_info.items():
        whole_value = whole_info[name]
        # 此时的结果应该是不同的
        if sample_value == whole_value:
            logging.warning(
                f"[unnormalized] {name} should be not equal: {sample_value} vs {whole_value}"
            )
        else:
            logging.info(f"[unnormalized] {name} passed!")


def main():
    pred_dir = "test_data/preds"
    mask_dir = "test_data/masks"
    pred_files = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir)])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    compare_normalized(pred_files, mask_files)
    compare_unnormalized(pred_files, mask_files)


if __name__ == "__main__":
    main()
