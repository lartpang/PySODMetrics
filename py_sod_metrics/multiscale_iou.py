import numpy as np
from scipy import ndimage

from .utils import TYPE, get_adaptive_threshold, trapezoid, validate_and_normalize_input


class MSIoU:
    """Multi-Scale Intersection over Union (MSIoU) metric.

    This implements the MSIoU metric which evaluates segmentation quality at multiple scales by comparing edge maps. It addresses the limitation of traditional IoU which struggles with fine structures in segmentation results.


    ```
    @inproceedings{MSIoU,
        title = {Multiscale IOU: A Metric for Evaluation of Salient Object Detection with Fine Structures},
        author = {Ahmadzadeh, Azim and Kempton, Dustin J. and Chen, Yang and Angryk, Rafal A.},
        booktitle = ICIP,
        year = {2021},
    }
    ```
    """

    def __init__(self, with_dynamic: bool, with_adaptive: bool, *, with_binary: bool = False, num_levels: int = 10):
        """Initialize the MSIoU evaluator.

        Args:
            with_dynamic (bool, optional): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool, optional): Record adaptive results for adp version.
            with_binary (bool, optional): Record binary results for binary version.
            num_levels (int, optional): Number of scales in [1, 63], using the cell sizes [2^0, 2^1, ..., 2^(num_levels - 1)]. A single level returns the edge IoU at the original resolution. Defaults to 10 (1 to 512) as in the original implementation.
        """
        # 63: 2^62 is the largest cell size that fits in int64.
        if not (isinstance(num_levels, (int, np.integer)) and 1 <= num_levels <= 63):
            raise ValueError(f"num_levels must be an integer in [1, 63], got {num_levels!r}.")

        self.dynamic_results = [] if with_dynamic else None
        self.adaptive_results = [] if with_adaptive else None
        self.binary_results = [] if with_binary else None

        # The values of this collection determines the resolutions based on which MIoU is computed.
        self.cell_sizes = np.power(2, np.arange(num_levels, dtype=np.int64))

    def get_edge(self, mask: np.ndarray):
        """Edge detection based on the `scipy.ndimage.sobel` function.

        :param mask: a binary mask of an object whose edges are of interest.
        :return: a binary mask of 1's as edges and 0's as background.
        """
        # NOTE: Cast to float64 first. For a bool mask, `np.hypot` of the bool sobel outputs returns float16, in which the subsequent pixel counting (`.sum()`) loses precision beyond 2048 and becomes platform-dependent.
        mask = np.asarray(mask, dtype=TYPE)
        sx = ndimage.sobel(mask, axis=0, mode="constant")
        sy = ndimage.sobel(mask, axis=1, mode="constant")
        sob = np.hypot(sx, sy)
        # sob[sob > 0] = 1
        return (sob > 0).astype(TYPE)

    def shrink_by_grid(self, image: np.ndarray, cell_size: int) -> np.ndarray:
        """Shrink the image by checking for any non-zero values within grid cells.

        Performs box-counting after applying zero padding if the image dimensions
        are not perfectly divisible by the cell size.

        :param image: The input binary image (edges).
        :param cell_size: The size of the grid cells.
        :return: A shrunk binary image where each pixel represents a grid cell.
        """
        if cell_size <= 0:
            raise ValueError("Cell size must be a positive integer")

        if cell_size > 1:
            h, w = image.shape[:2]
            # If the cell size reaches the image size along an axis, there is only one cell along that axis.
            # Using the image size as the cell size there gives the same result without padding the image up to `cell_size`, which would explode the memory for large cell sizes (e.g., 2^15 for 16 levels).
            cell_h = min(int(cell_size), h)
            cell_w = min(int(cell_size), w)

            # Calculate padding sizes to make dimensions divisible by the cell size
            pad_h = (cell_h - h % cell_h) % cell_h
            pad_w = (cell_w - w % cell_w) % cell_w

            # Apply padding if necessary
            if pad_h > 0 or pad_w > 0:
                # Padding is added to the top and left edges.
                image = np.pad(image, ((pad_h, 0), (pad_w, 0)), mode="constant", constant_values=0)

            # Reshape and check for any non-zero value within each cell
            h, w = image.shape[:2]
            image = image.reshape(h // cell_h, cell_h, w // cell_w, cell_w)
            # Use any() for efficiency, but maintain the exact original data type to avoid precision drift in subsequent sums
            return image.any(axis=(1, 3)).astype(image.dtype)

        # image[image > 0] = 1
        return (image > 0).astype(image.dtype)

    def multi_scale_iou(self, pred_edge: np.ndarray, gt_edge: np.ndarray, gt_shrunk_cache: list = None) -> list:
        """Calculate Multi-Scale IoU.

        Args:
            pred_edge (np.ndarray): edge map of pred
            gt_edge (np.ndarray): edge map of gt
            gt_shrunk_cache (list, optional): precomputed GT shrunk results.

        Returns:
            list: ratios
        """
        # Calculate IoU ratios at different scales
        ratios = []
        for i, cell_size in enumerate(self.cell_sizes):
            # Shrink both prediction and ground truth edges
            shrunk_pred_edge = self.shrink_by_grid(pred_edge, cell_size=cell_size)

            if gt_shrunk_cache is not None:
                shrunk_gt_edge = gt_shrunk_cache[i]
            else:
                shrunk_gt_edge = self.shrink_by_grid(gt_edge, cell_size=cell_size)

            # Calculate IoU with smoothing to prevent division by zero
            numerator = np.logical_and(shrunk_pred_edge, shrunk_gt_edge).sum() + 1
            # Only consider ground truth for denominator
            denominator = shrunk_gt_edge.sum() + 1
            ratios.append(numerator / denominator)
        return ratios

    def binarizing(self, pred_bin: np.ndarray, gt_edge: np.ndarray, gt_shrunk_cache: list = None) -> list:
        """Calculate Multi-Scale IoU based on dynamically thresholding.

        Args:
            pred_bin (np.ndarray): binarized pred
            gt_edge (np.ndarray): gt binarized by 128
            gt_shrunk_cache (list, optional): precomputed GT shrunk results.

        Returns:
            np.ndarray: areas under the curve
        """
        pred_edge = self.get_edge(pred_bin)
        ratios = self.multi_scale_iou(pred_edge, gt_edge, gt_shrunk_cache=gt_shrunk_cache)  # 10

        if len(ratios) == 1:
            # Single-scale fallback: there is no curve to integrate, so return the edge IoU itself.
            return ratios[0]

        # Calculate area under the curve using trapezoidal rule
        return trapezoid(y=ratios, dx=1 / (len(self.cell_sizes) - 1))

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Calculate the Multi-Scale IoU for a single prediction-ground truth pair.

        This method first extracts edges from both prediction and ground truth,
        then computes IoU ratios at multiple scales defined by self.cell_sizes.
        Finally, it calculates the area under the curve of these ratios.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.

        Returns:
            The MSIoU score for the given pair (float between 0 and 1).
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        # Calculate MSIoU for this pair and store the result
        gt_edge = self.get_edge(gt)

        if self.dynamic_results is not None:
            gt_shrunk_cache = [self.shrink_by_grid(gt_edge, cell_size=cell_size) for cell_size in self.cell_sizes]

            results = []
            _pred = (pred * 255).astype(np.uint8)
            changed_thresholds = set(np.unique(_pred).astype(int) + 1)
            for threshold in range(256):
                if threshold == 0 or threshold in changed_thresholds:
                    results.append(self.binarizing(_pred >= threshold, gt_edge, gt_shrunk_cache))
                else:
                    results.append(results[-1])
            self.dynamic_results.append(results)

        if self.adaptive_results is not None:
            adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
            results = self.binarizing(pred >= adaptive_threshold, gt_edge)
            self.adaptive_results.append(results)

        if self.binary_results is not None:
            self.binary_results.append(self.binarizing(pred > 0.5, gt_edge))

    def get_results(self) -> dict:
        """Return the results about MSIoU.

        Calculates the mean of all stored MSIoU values from previous calls to step().

        :return: Dictionary with key 'msiou' and the mean MSIoU value.
        :raises: ValueError if no samples have been processed.
        """
        results = {}
        if self.dynamic_results is not None:
            results["dynamic"] = np.mean(np.array(self.dynamic_results, dtype=TYPE), axis=0)
        if self.adaptive_results is not None:
            results["adaptive"] = np.mean(np.array(self.adaptive_results, dtype=TYPE))
        if self.binary_results is not None:
            results["binary"] = np.mean(np.array(self.binary_results, dtype=TYPE))
        return results
