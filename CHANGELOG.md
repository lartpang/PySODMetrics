# CHANGELOG

## [1.4.3] - 2025-5-8

- Migrate to modern PyPI publishing by configuring `pyproject.toml` and `python-publish.yml`.
- Update the formatter and linter tools to `ruff`.
- Update the documentation information for the functions in `py_sod_metrics/fmeasurev2.py` and `py_sod_metrics/multiscale_iou.py`.
- Optimize the code in `py_sod_metrics/multiscale_iou.py`.

## [1.4.3.1] - 2025-5-8

- [FEATURE] Add `binary`, `dinamic`, and `adaptive` modes for `py_sod_metrics/multiscale_iou.py`.
- [UPDATE] Update `examples/test_metrics.py` to support `binary`, `dinamic`, and `adaptive` modes of `MSIoU`.
- [NOTE] The current implementation of the dynamic mode for `MSIoU` relies on the for loop, so it currently runs less efficiently.

## [1.4.4] - 2025-5-9

- [FEATURE] Add `normalize` parameter to `py_sod_metrics/sod_metrics.py`.
- [UPDATE] Update a unified function `validate_and_normalize_input` to validate and normalize the input data.

## [1.7.0] - 2026-11-17

- [FIX] `MSIoU`: Edge maps of boolean masks were float16, so edge pixel counts lost precision beyond 2048 and the results differed across platforms. Edge maps are now float64 and the counting is exact. **MSIoU values change slightly (around the 6th decimal place).** The new baselines are recorded under `v1_7_0` in `examples/version_performance.json`.
- [FIX] `HumanCorrectionEffortMeasure`: `cv2.approxPolyDP` changed its results since OpenCV 4.13, making HCE depend on the OpenCV version (e.g., 74.33 instead of 73.67 on `examples/test_data`). It is replaced by `approx_poly_dp` in `py_sod_metrics/utils.py`, which reproduces the behavior of OpenCV <= 4.12, so HCE values are unchanged and no longer depend on the OpenCV version.
- [FIX] `Smeasure`: Return a valid score instead of NaN when the foreground centroid lies on the last row/column.
- [FIX] `SizeInvarianceMAE` / `SizeInvarianceFmeasureV2`: Support ground truths that are entirely foreground, target bounding boxes covering the whole image, and more than 63 targets. The bitwise helpers `encode_bboxwise_tgts_bitwise` and `get_kth_bit` are replaced by `get_bboxwise_tgt_masks`.
- [UPDATE] `MSIoU`: `num_levels` is no longer limited to 10. The k-th level uses the grid cell size 2^k, i.e., the cell sizes are [2^0, ..., 2^(num_levels - 1)]. The default `num_levels=10` still gives 1 to 512 as in the [original implementation](https://bitbucket.org/gsudmlab/multiscale_iou) (note that values in [2, 9] previously sampled cell sizes evenly from 1 to 512). `num_levels=1` falls back to the single-scale edge IoU. Cell sizes larger than the image no longer pad the image up to the cell size.
- [FIX] `MSIoU`: Validate `num_levels` (an integer in [1, 63]) and replace the deprecated `np.trapz` with `np.trapezoid` (falling back to `np.trapz` on NumPy < 2.0).
