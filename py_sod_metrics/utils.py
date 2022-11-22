import numpy as np

# the different implementation of epsilon (extreme min value) between numpy and matlab
EPS = np.spacing(1)
TYPE = np.float64


def prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    A numpy-based function for preparing ``pred`` and ``gt``.

    - for ``pred``, it looks like ``mapminmax(im2double(...))`` of matlab;
    - ``gt`` will be binarized by 128.

    :param pred: prediction
    :param gt: mask
    :return: pred, gt
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.

    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)
