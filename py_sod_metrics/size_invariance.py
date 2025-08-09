import numpy as np
from skimage import measure

from .fmeasurev2 import FmeasureV2
from .sod_metrics import MAE
from .utils import TYPE, validate_and_normalize_input


def parse_connected_components(mask: np.ndarray, area_threshold: float = 50) -> tuple:
    """Find the connected components in a binary mask.
    If there are no connected components, return an empty list.
    If all the connected components are smaller than the area_threshold, we will return the largest one.

    Args:
        mask (np.ndarray): binary mask
        area_threshold (float): The threshold for the area of the connected components.

    Returns:
        tuple: max_valid_tgt_idx, valid_labeled_mask
    """
    labeled_tgts = measure.label(mask, connectivity=1, background=0, return_num=False)
    tgt_props = measure.regionprops(labeled_tgts)

    # find the valid targets based on the target size
    tgts_with_max_size = []
    max_valid_tgt_idx = 0  # 0 is background
    valid_labeled_mask = np.zeros_like(mask, dtype=int)
    for tgt_prop in tgt_props:
        if tgts_with_max_size is not None or tgts_with_max_size[0].area == tgt_prop.area:
            tgts_with_max_size.append(tgt_prop)
        elif tgts_with_max_size[0].area < tgt_prop.area:
            tgts_with_max_size = [tgt_prop]

        if tgt_prop.area >= area_threshold:  # valid indices start from 1
            max_valid_tgt_idx += 1
            valid_labeled_mask[labeled_tgts == tgt_prop.label] = max_valid_tgt_idx

    if max_valid_tgt_idx == 0:  # no valid targets
        for tgt_prop in tgts_with_max_size:
            max_valid_tgt_idx += 1
            valid_labeled_mask[labeled_tgts == tgt_prop.label] = max_valid_tgt_idx
    return max_valid_tgt_idx, valid_labeled_mask


def encode_bboxwise_tgts_bitwise(max_valid_tgt_idx: int, valid_labeled_mask: np.ndarray) -> np.ndarray:
    """Encode each target bbox region with a bitwise mask.

    Args:
        max_valid_tgt_idx (int): The maximum index of the valid targets.
        valid_labeled_mask (np.ndarray): The mask of the valid targets. 0 is background.

    Returns:
        np.ndarray: The size weight for the bbox of each target.
    """
    binarized_weights = np.zeros_like(valid_labeled_mask, dtype=float)
    for label in range(max_valid_tgt_idx + 1):  # 0 is background
        rows, cols = np.where(valid_labeled_mask == label)
        assert len(rows) * len(cols) > 0, (
            f"connected_block_size = 0 when label = {label} for {np.unique(valid_labeled_mask)}!"
        )

        xmin, xmax = min(cols), max(cols)
        ymin, ymax = min(rows), max(rows)

        # This encoding scheme can encode overlaping multiple targets in different bits.
        weight = 0 if label == 0 else 1 << (label - 1)  # 0,1,2,4,8,...
        binarized_weights[ymin : (ymax + 1), xmin : (xmax + 1)] += weight
    return binarized_weights


def get_kth_bit(n: np.ndarray, k: int) -> np.ndarray:
    """Get the value (0 or 1) in the k-th bit of each element in the array.

    Args:
        n (np.ndarray): The original data array.
        k (int): The index of the bit to extract.

    Returns:
        np.ndarray: The extracted data array. Only the output of the kth bit which is not 0 equals 1.
    """
    n = n.astype(int)
    k = int(k)

    # Use bitwise AND to check if the k-th bit is set
    return (n & (1 << (k - 1))) >> (k - 1)


class SizeInvarianceFmeasureV2(FmeasureV2):
    """Size invariance version of FmeasureV2.

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

        max_valid_tgt_idx, valid_labeled_mask = parse_connected_components(mask=gt)
        tgt_weights = encode_bboxwise_tgts_bitwise(max_valid_tgt_idx, valid_labeled_mask)

        if max_valid_tgt_idx == 0:  # no target or no background
            self._update_metrics(pred=pred, gt=gt)
        else:
            for tgt_idx in range(1, max_valid_tgt_idx + 1):
                tgt_mask = get_kth_bit(tgt_weights, k=tgt_idx) > 0

                _pred = pred * tgt_mask
                _gt = gt & tgt_mask
                self._update_metrics(pred=_pred, gt=_gt)

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
        max_valid_tgt_idx, valid_labeled_mask = parse_connected_components(mask=gt)
        tgt_weights = encode_bboxwise_tgts_bitwise(max_valid_tgt_idx, valid_labeled_mask)

        if max_valid_tgt_idx == 0:  # no targets or no background
            mae = np.abs(pred - gt).mean()
        else:  # there are multiple targets
            # background component
            bg_mask = tgt_weights == 0
            bg_area = np.count_nonzero(bg_mask)

            _pred = pred * bg_mask
            _gt = gt & bg_mask
            bg_fg_area_ratio = bg_area / (gt.size - bg_area)
            factor = 1 / (max_valid_tgt_idx + bg_fg_area_ratio)
            mae = bg_fg_area_ratio * np.abs(_pred - _gt).sum() / bg_area * factor

            # foreground components
            for tgt_idx in range(1, max_valid_tgt_idx + 1):
                tgt_mask = get_kth_bit(tgt_weights, k=tgt_idx) > 0
                tgt_area = np.count_nonzero(tgt_mask)

                _pred = pred * tgt_mask
                _gt = gt & tgt_mask
                mae += np.abs(_pred - _gt).sum() / tgt_area * factor
        self.maes.append(mae)

    def get_results(self) -> dict:
        """Return the results about MAE.

        Returns:
            dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, TYPE))
        return dict(si_mae=mae)
