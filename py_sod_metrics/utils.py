import numpy as np

# the different implementation of epsilon (extreme min value) between numpy and matlab
EPS = np.spacing(1)
TYPE = np.float64


def validate_and_normalize_input(pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
    """Performs input validation and normalization."""
    # Validate input shapes
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch between prediction ({pred.shape}) and ground truth ({gt.shape})"
        )

    # Handle normalization
    if normalize:
        pred, gt = prepare_data(pred, gt)
    else:
        # Validate prediction data type and range
        if pred.dtype not in (np.float32, np.float64):
            raise TypeError(f"Prediction array must be float32 or float64, got {pred.dtype}")
        if not (0 <= pred.min() and pred.max() <= 1):
            raise ValueError("Prediction values must be in range [0, 1]")
        # Validate ground truth type
        if gt.dtype != bool:
            raise TypeError(f"Ground truth must be boolean, got {gt.dtype}")

    return pred, gt


def prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """A numpy-based function for preparing `pred` and `gt`.

    - for `pred`, it looks like `mapminmax(im2double(...))` of matlab;
    - `gt` will be binarized by 128.

    Args:
        pred (np.uint8): Prediction, gray scale image.
        gt (np.uint8): Ground truth, gray scale image.

    Returns:
        tuple: pred (np.float64), gt (bool)
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """Return an adaptive threshold, which is equal to twice the mean of `matrix`.

    Args:
        matrix (np.ndarray): a data array
        max_value (float, optional): the upper limit of the threshold. Defaults to 1.

    Returns:
        float: `min(2 * matrix.mean(), max_value)`
    """
    return min(2 * matrix.mean(), max_value)
