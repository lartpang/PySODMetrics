import numpy as np
from scipy import ndimage

from .utils import TYPE


class MSIoU:
    def __init__(self):
        """
        Multi-Scale Intersection over Union (MSIoU) metric.

        ::

            @inproceedings{MSIoU,
                title = {Multiscale IOU: A Metric for Evaluation of Salient Object Detection with Fine Structures},
                author = {Ahmadzadeh, Azim and Kempton, Dustin J. and Chen, Yang and Angryk, Rafal A.},
                booktitle = ICIP,
                year = {2021},
            }
        """
        # The values of this collection determines the resolutions based on which MIoU is computed.
        # It is set as the original implementation
        self.cell_sizes = np.power(2, np.linspace(0, 9, num=10, dtype=int))
        self.msious = []

    def get_edge(self, mask: np.ndarray):
        """Edge detection based on the `scipy.ndimage.sobel` function.

        :param mask: a binary mask of an object whose edges are of interest.
        :return: a binary mask of 1's as edges and 0's as background.
        """
        sx = ndimage.sobel(mask, axis=0, mode="constant")
        sy = ndimage.sobel(mask, axis=1, mode="constant")
        sob = np.hypot(sx, sy)
        sob[sob > 0] = 1
        return sob

    def shrink_by_grid(self, image: np.ndarray, cell_size: int) -> np.ndarray:
        """Shrink the image by summing values within grid cells.

        Performs box-counting after applying zero padding if the image dimensions
        are not perfectly divisible by the cell size.

        :param image: The input binary image (edges).
        :param cell_size: The size of the grid cells.
        :return: A shrunk binary image where each pixel represents a grid cell.
        """
        if cell_size <= 0:
            raise ValueError("Cell size must be a positive integer")

        h, w = image.shape[:2]

        # Calculate padding sizes to make dimensions divisible by cell_size
        pad_h = (cell_size - h % cell_size) % cell_size
        pad_w = (cell_size - w % cell_size) % cell_size

        # Apply padding if necessary
        if pad_h > 0 or pad_w > 0:
            # Padding is added to the top and left edges.
            image = np.pad(image, ((pad_h, 0), (pad_w, 0)), mode="constant", constant_values=0)
        # Reshape and sum within each cell

        h, w = image.shape[:2]
        image = image.reshape(h // cell_size, cell_size, w // cell_size, cell_size)
        image = image.sum(axis=(1, 3))
        image[image > 0] = 1
        return image

    def cal_msiou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate the Multi-Scale IoU for a single prediction-ground truth pair.

        This method first extracts edges from both prediction and ground truth,
        then computes IoU ratios at multiple scales defined by self.cell_sizes.
        Finally, it calculates the area under the curve of these ratios.

        :param pred: Binary prediction mask.
        :param gt: Binary ground truth mask.
        :return: The MSIoU score for the given pair (float between 0 and 1).
        """
        # Extract edges from both prediction and ground truth
        pred_edge = self.get_edge(pred)
        gt_edge = self.get_edge(gt)

        # Calculate IoU ratios at different scales
        ratios = []
        for cell_size in self.cell_sizes:
            # Shrink both prediction and ground truth edges
            s_pred = self.shrink_by_grid(pred_edge, cell_size=cell_size)
            s_gt = self.shrink_by_grid(gt_edge, cell_size=cell_size)

            # Calculate IoU with smoothing to prevent division by zero
            numerator = np.logical_and(s_pred, s_gt).sum() + 1
            # Only consider ground truth for denominator
            denominator = s_gt.sum() + 1
            ratios.append(numerator / denominator)

        # Calculate area under the curve using trapezoidal rule
        if len(self.cell_sizes) > 1:
            msiou = np.trapz(y=ratios, dx=1 / (len(self.cell_sizes) - 1))
        else:
            # Handle edge case with only one cell size
            msiou = ratios[0]

        return msiou

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Process one prediction-ground truth pair.

        Binarize predictions and ground truth using a threshold of 128, calculates MSIoU,
        and stores the result for later aggregation.

        :param pred: Grayscale prediction map (0-255).
        :param gt: Grayscale ground truth map (0-255).
        """
        # Validate input
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

        # Binarize input arrays (assumes grayscale images with values in [0, 255])
        gt_bin = gt > 128
        pred_bin = pred > 128

        # Calculate MSIoU for this pair and store the result
        msiou = self.cal_msiou(pred_bin, gt_bin)
        self.msious.append(msiou)

    def get_results(self) -> dict:
        """Return the results about MSIoU.

        Calculates the mean of all stored MSIoU values from previous calls to step().

        :return: Dictionary with key 'msiou' and the mean MSIoU value.
        :raises: ValueError if no samples have been processed.
        """
        if not self.msious:
            raise ValueError("No samples have been processed. Call step() first.")

        # Calculate mean MSIoU across all processed samples
        msiou = np.mean(np.array(self.msious, TYPE))
        return {"msiou": msiou}
