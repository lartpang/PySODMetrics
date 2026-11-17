import numpy as np

# the different implementation of epsilon (extreme min value) between numpy and matlab
EPS = np.spacing(1)
TYPE = np.float64

# `np.trapz` is deprecated since NumPy 2.0 in favor of `np.trapezoid`, which does not exist before NumPy 2.0.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


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


def approx_poly_dp(curve: np.ndarray, epsilon: float, closed: bool) -> np.ndarray:
    """Approximate a polygonal curve with the Ramer-Douglas-Peucker algorithm.

    This reproduces `cv2.approxPolyDP` of OpenCV <= 4.12 for integer points, which measures the distance from a point to the infinite line through the two ends of the current segment. OpenCV >= 4.13 changes this behavior and thus the results of metrics like HCE, so we implement it here to keep the results independent of the OpenCV version.

    Args:
        curve (np.ndarray): Integer points of the curve with shape (N, 2) or (N, 1, 2) (OpenCV contour format).
        epsilon (float): The maximum distance between the original curve and its approximation.
        closed (bool): Whether the curve is closed.

    Returns:
        np.ndarray: The approximated curve with shape (M, 1, 2) and dtype int32.
    """
    src = np.asarray(curve).reshape(-1, 2)
    if src.size and not np.issubdtype(src.dtype, np.integer):
        raise TypeError(f"Only integer points are supported, got {src.dtype}.")
    src = src.astype(np.int64)
    count = len(src)
    if count == 0:
        return np.zeros((0, 1, 2), dtype=np.int32)

    eps = epsilon * epsilon
    stack = []  # slices (start, end) of the curve to be processed, the end point is excluded from the output
    dst = []
    right_start = 0
    is_closed = closed
    init_iters = 3
    le_eps = False
    start_pt = None

    if not is_closed:
        if np.array_equal(src[0], src[-1]):
            # an open curve whose ends coincide is processed as a closed one
            is_closed = True
            init_iters = 1
        else:
            stack.append((0, count - 1))

    if is_closed:
        # 1. Find approximately two farthest points of the curve as the initial split.
        pos = 0
        for _ in range(init_iters):
            pos = (pos + right_start) % count
            start_pt = src[pos]
            dists = np.sum((src[(pos + np.arange(1, count)) % count] - start_pt) ** 2, axis=1)
            max_dist = dists.max(initial=0)
            if max_dist > 0:  # the first farthest point, as OpenCV only updates it on strictly larger distances
                right_start = int(np.argmax(dists)) + 1
            le_eps = max_dist <= eps

        # 2. Initialize the stack.
        if not le_eps:
            slice_start = pos
            right_start = (right_start + slice_start) % count
            stack.append((right_start, slice_start))
            stack.append((slice_start, right_start))
        else:
            dst.append(start_pt)

    # 3. Recursively split the slices at the farthest points (depth-first, the left part first).
    while stack:
        slice_start, slice_end = stack.pop()
        start_pt = src[slice_start]
        end_pt = src[slice_end]
        num_inner = (slice_end - slice_start - 1) % count  # number of points strictly between the two ends
        if num_inner > 0:
            dx, dy = end_pt - start_pt
            if dx == 0 and dy == 0:
                raise ValueError("The two ends of a slice coincide, which is not supported (as in OpenCV).")
            inner_idx = (slice_start + 1 + np.arange(num_inner)) % count
            inner = src[inner_idx]
            # the distance to the line through the two ends, scaled by the length of the segment
            dists = np.abs((inner[:, 1] - start_pt[1]) * dx - (inner[:, 0] - start_pt[0]) * dy)
            max_idx = int(np.argmax(dists))
            max_dist = float(dists[max_idx])  # compared in double precision as in OpenCV
            le_eps = max_dist * max_dist <= eps * float(dx * dx + dy * dy)
            if not le_eps:
                right_start = int(inner_idx[max_idx])
        else:
            le_eps = True

        if le_eps:
            dst.append(start_pt)
        else:
            stack.append((right_start, slice_end))
            stack.append((slice_start, right_start))

    if not is_closed:
        dst.append(src[-1])

    # 4. Final clean-up: remove the extra points on the (almost) straight lines.
    is_closed = closed
    count = new_count = len(dst)
    pos = count - 1 if is_closed else 0

    def read(p):
        return dst[p], (p + 1) % count

    start_pt, pos = read(pos)
    wpos = pos
    pt, pos = read(pos)
    i = 0 if is_closed else 1
    while i < count - (0 if is_closed else 1) and new_count > 2:
        end_pt, pos = read(pos)
        dx, dy = end_pt - start_pt
        dist = float(abs((pt[0] - start_pt[0]) * dy - (pt[1] - start_pt[1]) * dx))
        successive_inner_product = (pt[0] - start_pt[0]) * (end_pt[0] - pt[0]) + (pt[1] - start_pt[1]) * (
            end_pt[1] - pt[1]
        )
        if (
            dist * dist <= 0.5 * eps * float(dx * dx + dy * dy)
            and dx != 0
            and dy != 0
            and successive_inner_product >= 0
        ):
            new_count -= 1
            dst[wpos] = start_pt = end_pt
            wpos = (wpos + 1) % count
            pt, pos = read(pos)
            i += 2
            continue
        dst[wpos] = start_pt = pt
        wpos = (wpos + 1) % count
        pt = end_pt
        i += 1

    if not is_closed:
        dst[wpos] = pt
    return np.asarray(dst[:new_count], dtype=np.int32).reshape(-1, 1, 2)


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """Return an adaptive threshold, which is equal to twice the mean of `matrix`.

    Args:
        matrix (np.ndarray): a data array
        max_value (float, optional): the upper limit of the threshold. Defaults to 1.

    Returns:
        float: `min(2 * matrix.mean(), max_value)`
    """
    return min(2 * matrix.mean(), max_value)
