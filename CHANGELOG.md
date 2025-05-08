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
