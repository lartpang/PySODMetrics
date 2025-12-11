import numpy as np

# the different implementation of epsilon (extreme min value) between numpy and matlab
EPS = np.spacing(1)
TYPE = np.float64


def validate_and_normalize_input(pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
    """Validate and optionally normalize prediction and ground truth inputs.

    This function ensures that prediction and ground truth arrays have compatible shapes and appropriate data types. When normalization is enabled, it converts inputs to the standard format required by the predefined metrics (pred in [0, 1] as float, gt as boolean).

    Args:
        pred (np.ndarray): Prediction array. If `normalize=True`, should be uint8 grayscale image (0-255). If `normalize=False`, should be float32/float64 in range [0, 1].
        gt (np.ndarray): Ground truth array. If `normalize=True`, should be uint8 grayscale image (0-255). If `normalize=False`, should be boolean array.
        normalize (bool, optional): Whether to normalize the input data using prepare_data(). Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pred (np.ndarray): Normalized prediction as float64 in range [0, 1].
            - gt (np.ndarray): Normalized ground truth as boolean array.

    Raises:
        ValueError: If prediction and ground truth shapes don't match, or if prediction values are outside [0, 1] range when normalize=False.
        TypeError: If data types are invalid when normalize=False (pred must be float32/float64, gt must be boolean).
    """
    # Validate input shapes
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch between prediction ({pred.shape}) and ground truth ({gt.shape})")

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
    """Convert and normalize prediction and ground truth data.

    - For predictions, mimics MATLAB's `mapminmax(im2double(...))`.
    - For ground truth, applies binary thresholding at 128.

    Args:
        pred (np.ndarray): Prediction grayscale image, uint8 type with values in [0, 255].
        gt (np.ndarray): Ground truth grayscale image, uint8 type with values in [0, 255].

    Returns:
        tuple: A tuple containing:
            - pred (np.ndarray): Normalized prediction as float64 in range [0, 1].
            - gt (np.ndarray): Binary ground truth as boolean array.
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
