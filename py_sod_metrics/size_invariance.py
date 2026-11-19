import numpy as np
from scipy import ndimage
from skimage import measure

from .fmeasurev2 import FmeasureV2
from .sod_metrics import MAE
from .utils import TYPE, validate_and_normalize_input


def _label_targets(mask: np.ndarray, area_threshold: float = 50) -> tuple:
    labeled_tgts = measure.label(mask, connectivity=1, background=0, return_num=False)
    areas = np.bincount(labeled_tgts.ravel())[1:]  # 0 is background
    target_labels = np.flatnonzero(areas >= area_threshold)
    if target_labels.size == 0 and areas.size > 0:
        # Keep all targets tied for the largest area.
        target_labels = np.flatnonzero(areas == areas.max())
    return labeled_tgts, target_labels + 1


def _get_target_boxes(mask: np.ndarray) -> list:
    labeled_tgts, target_labels = _label_targets(mask)
    if target_labels.size == 0:
        return []
    boxes = ndimage.find_objects(labeled_tgts)
    return [boxes[label - 1] for label in target_labels]


def parse_connected_components(mask: np.ndarray, area_threshold: float = 50) -> tuple:
    """Find the connected components in a binary mask.

    1. If there are no connected components, return an empty list.
    2. If all the connected components are smaller than the area_threshold, we will return the largest one.

    Args:
        mask (np.ndarray): binary mask
        area_threshold (float): The threshold for the area of the connected components.

    Returns:
        tuple: max_valid_tgt_idx, valid_labeled_mask
    """
    labeled_tgts, target_labels = _label_targets(mask, area_threshold)
    label_map = np.zeros(labeled_tgts.max(initial=0) + 1, dtype=int)
    label_map[target_labels] = np.arange(1, target_labels.size + 1)
    return target_labels.size, label_map[labeled_tgts]


def get_bboxwise_tgt_masks(max_valid_tgt_idx: int, valid_labeled_mask: np.ndarray) -> list:
    """Generate a binary mask covering the bounding box of each valid target.

    Args:
        max_valid_tgt_idx (int): The maximum index of the valid targets.
        valid_labeled_mask (np.ndarray): The mask of the valid targets. 0 is background.

    Returns:
        list: `max_valid_tgt_idx` boolean masks, the i-th one covering the bbox of the target with label i+1.
    """
    tgt_masks = []
    for label in range(1, max_valid_tgt_idx + 1):  # 0 is background
        rows, cols = np.nonzero(valid_labeled_mask == label)
        if rows.size == 0:
            raise ValueError(f"Target with label {label} is empty in {np.unique(valid_labeled_mask)}!")

        tgt_mask = np.zeros(valid_labeled_mask.shape, dtype=bool)
        tgt_mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] = True
        tgt_masks.append(tgt_mask)
    return tgt_masks


class SizeInvarianceFmeasureV2(FmeasureV2):
    """Size invariance version of FmeasureV2.

    This provides size-invariant versions of standard SOD metrics that address the imbalance problem in multi-object salient object detection. Traditional metrics can be biased toward larger objects, while size-invariant metrics ensure fair evaluation across objects of different sizes.

    ```
    @inproceedings{SizeInvarianceVariants,
        title = {Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection},
        author = {Feiran Li and Qianqian Xu and Shilong Bao and Zhiyong Yang and Runmin Cong and Xiaochun Cao and Qingming Huang},
        booktitle = ICML,
        year = {2024}
    }
    ```
    """

    def _update_metrics(self, pred: np.ndarray, gt: np.ndarray):
        FG = np.count_nonzero(gt)  # 真实前景, FG=(TPs+FNs)
        BG = gt.size - FG  # 真实背景, BG=(TNs+FPs)

        dynamical_tpfptnfn = None
        adaptive_tpfptnfn = None
        binary_tpfptnfn = None
        for handler_name, handler in self._metric_handlers.items():
            if handler.dynamic_results is not None:
                if dynamical_tpfptnfn is None:
                    dynamical_tpfptnfn = self.dynamically_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)
                tgt_result = handler(**dynamical_tpfptnfn)
                if handler.sample_based:  # is not None
                    if not handler.dynamic_results or not isinstance(
                        handler.dynamic_results[-1], list
                    ):  # is not [] or not contain list
                        handler.dynamic_results.append([])
                    handler.dynamic_results[-1].append(tgt_result)
                else:
                    handler.dynamic_results.append(tgt_result)

            if handler.adaptive_results is not None:
                if adaptive_tpfptnfn is None:
                    adaptive_tpfptnfn = self.adaptively_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)
                tgt_result = handler(**adaptive_tpfptnfn)
                if not handler.adaptive_results or not isinstance(handler.adaptive_results[-1], list):
                    handler.adaptive_results.append([])
                handler.adaptive_results[-1].append(tgt_result)

            if handler.binary_results is not None:
                if binary_tpfptnfn is None:
                    # `pred > 0.5`: Simulating the effect of the `argmax` function.
                    binary_tpfptnfn = self.get_statistics(binary=pred > 0.5, gt=gt, FG=FG, BG=BG)

                if handler.sample_based:
                    tgt_result = handler(**binary_tpfptnfn)
                    if not handler.binary_results or not isinstance(handler.binary_results[-1], list):
                        handler.binary_results.append([])
                    handler.binary_results[-1].append(tgt_result)
                else:  # will average over all targets from all samples
                    tgt_result = binary_tpfptnfn
                    handler.binary_results["tp"] += tgt_result["tp"]
                    handler.binary_results["fp"] += tgt_result["fp"]
                    handler.binary_results["tn"] += tgt_result["tn"]
                    handler.binary_results["fn"] += tgt_result["fn"]

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metrics for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        if not self._metric_handlers:  # 没有添加metric_handler
            raise ValueError("Please add your metric handler before using `step()`.")

        pred, gt = validate_and_normalize_input(pred, gt, normalize=normalize)

        target_boxes = _get_target_boxes(gt)

        if target_boxes:
            for target_box in target_boxes:
                target_pred = np.zeros_like(pred, order="C")
                target_gt = np.zeros_like(gt, order="C")
                target_pred[target_box] = pred[target_box]
                target_gt[target_box] = gt[target_box]
                self._update_metrics(pred=target_pred, gt=target_gt)
        else:
            self._update_metrics(pred=pred, gt=gt)

        # average over all targets in each sample
        for handler_name, handler in self._metric_handlers.items():
            if handler.dynamic_results is not None and handler.sample_based:
                tgt_results = handler.dynamic_results.pop()  # Tx256
                handler.dynamic_results.append(np.array(tgt_results, dtype=TYPE))  # Tx256

            if handler.adaptive_results is not None:
                tgt_results = handler.adaptive_results.pop()  # Tx1
                handler.adaptive_results.append(np.mean(np.array(tgt_results, dtype=TYPE)))  # 1

            if handler.binary_results is not None and handler.sample_based:
                tgt_results = handler.binary_results.pop()  # Tx1
                handler.binary_results.append(np.mean(np.array(tgt_results, dtype=TYPE)))  # 1

    def get_results(self) -> dict:
        """Return the results of the specific metric names.

        Returns:
            dict: All results corresponding to different metrics.
        """
        results = {}
        for handler_name, handler in self._metric_handlers.items():
            res = {}
            if handler.dynamic_results is not None:
                dynamic_results = handler.dynamic_results
                if handler.sample_based:  # N个T'x256
                    res["dynamic"] = dynamic_results
                else:  # N'x256 -> 256
                    res["dynamic"] = np.mean(np.array(dynamic_results, dtype=TYPE), axis=0)

            if handler.adaptive_results is not None:
                res["adaptive"] = np.mean(np.array(handler.adaptive_results, dtype=TYPE))  # 1

            if handler.binary_results is not None:
                binary_results = handler.binary_results
                if handler.sample_based:
                    res["binary"] = np.mean(np.array(binary_results, dtype=TYPE))  # 1
                else:
                    # NOTE: use `np.mean` to simplify output format (`array(123)` -> `123`)
                    res["binary"] = np.mean(handler(**binary_results))
            results[handler_name] = res
        return results


class SizeInvarianceMAE(MAE):
    """Size invariance version of MAE.

    ```
    @inproceedings{SizeInvarianceVariants,
        title = {Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection},
        author = {Feiran Li and Qianqian Xu and Shilong Bao and Zhiyong Yang and Runmin Cong and Xiaochun Cao and Qingming Huang},
        booktitle = ICML,
        year = {2024}
    }
    ```
    """

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize=normalize)
        target_boxes = _get_target_boxes(gt)

        if not target_boxes:
            mae = np.abs(pred - gt).mean()
        else:
            bg_mask = np.ones(gt.shape, dtype=bool)
            for target_box in target_boxes:
                bg_mask[target_box] = False
            bg_area = np.count_nonzero(bg_mask)

            # Sum zero-padded errors in image order to preserve floating-point results.
            errors = np.zeros_like(pred, order="C")
            mae = 0
            if bg_area:
                bg_fg_area_ratio = bg_area / (gt.size - bg_area)
                factor = 1 / (len(target_boxes) + bg_fg_area_ratio)
                np.subtract(pred, gt, out=errors, where=bg_mask)
                np.abs(errors, out=errors)
                mae = bg_fg_area_ratio * errors.sum() / bg_area * factor
                errors.fill(0)
            else:
                factor = 1 / len(target_boxes)

            # Match the scalar promotion of count_nonzero without allocating a target mask.
            area_type = type(bg_area)
            for target_box in target_boxes:
                target_errors = errors[target_box]
                np.subtract(pred[target_box], gt[target_box], out=target_errors)
                np.abs(target_errors, out=target_errors)
                mae += errors.sum() / area_type(target_errors.size) * factor
                target_errors.fill(0)
        self.maes.append(mae)

    def get_results(self) -> dict:
        """Return the results about MAE.

        Returns:
            dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, TYPE))
        return dict(si_mae=mae)
