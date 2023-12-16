<div align="center">
  <img src="https://github.com/lartpang/PySODMetrics/blob/main/images/logo.png?raw=true" alt="Logo" width="320" height="auto">
  </br>
  <h2>PySODMetrics: A simple and efficient implementation of SOD metrics</h2>
  <a href="./readme_zh.md"><img src="https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-blue"></a>
  <img src="https://img.shields.io/pypi/v/pysodmetrics">
  <img src="https://img.shields.io/pypi/dm/pysodmetrics?label=pypi%20downloads&logo=PyPI&logoColor=white">
  <img src="https://img.shields.io/github/last-commit/lartpang/PySODMetrics">
  <img src="https://img.shields.io/github/last-commit/lartpang/PySODMetrics">
  <img src="https://img.shields.io/github/release-date/lartpang/PySODMetrics">
</div>

## Introduction

A simple and efficient implementation of SOD metrics.

- Based on `numpy` and `scipy`
- Verification based on Fan's matlab code <https://github.com/DengPingFan/CODToolbox>
- The code structure is simple and easy to extend
- The code is lightweight and fast

Your improvements and suggestions are welcome.

### Related Projects

- [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit): A Python-based Evaluation Toolbox for Salient Object Detection and Camouflaged Object Detection

### Supported Metrics

| Metric                                    | Sample-based    | Whole-based | Related Class                         |
| ----------------------------------------- | --------------- | ----------- | ------------------------------------- |
| MAE                                       | soft            |             | `MAE`                                 |
| S-measure $S_{m}$                         | soft            |             | `Smeasure`                            |
| weighted F-measure ($F^{\omega}_{\beta}$) | soft            |             | `WeightedFmeasure`                    |
| Multi-Scale IoU                           | bin             |             | `MSIoU`                               |
| E-measure ($E_{m}$)                       | max,avg,adp     |             | `Emeasure`                            |
| F-measure (old) ($F_{beta}$)              | max,avg,adp     |             | `Fmeasure`                            |
| F-measure (new) ($F_{beta}$, $F_{1}$)     | max,avg,adp,bin | bin         | `FmeasureV2`+`FmeasureHandler`        |
| BER                                       | max,avg,adp,bin | bin         | `FmeasureV2`+`BERHandler`             |
| Dice                                      | max,avg,adp,bin | bin         | `FmeasureV2`+`DICEHandler`            |
| FPR                                       | max,avg,adp,bin | bin         | `FmeasureV2`+`FPRHandler`             |
| IoU                                       | max,avg,adp,bin | bin         | `FmeasureV2`+`IOUHandler`             |
| Kappa                                     | max,avg,adp,bin | bin         | `FmeasureV2`+`KappaHandler`           |
| Overall Accuracy                          | max,avg,adp,bin | bin         | `FmeasureV2`+`OverallAccuracyHandler` |
| Precision                                 | max,avg,adp,bin | bin         | `FmeasureV2`+`PrecisionHandler`       |
| Recall                                    | max,avg,adp,bin | bin         | `FmeasureV2`+`RecallHandler`          |
| Sensitivity                               | max,avg,adp,bin | bin         | `FmeasureV2`+`SensitivityHandler`     |
| Specificity                               | max,avg,adp,bin | bin         | `FmeasureV2`+`SpecificityHandler`     |
| TNR                                       | max,avg,adp,bin | bin         | `FmeasureV2`+`TNRHandler`             |
| TPR                                       | max,avg,adp,bin | bin         | `FmeasureV2`+`TPRHandler`             |

## Usage

The core files are in the folder `py_sod_metrics`.

- **[Latest, but may be unstable]** Install from the source code: `pip install git+https://github.com/lartpang/PySODMetrics.git`
- **[More stable]** Install from PyPI: `pip install pysodmetrics`

### Examples

- [examples/test_metrics.py](./examples/test_metrics.py)
- [examples/metric_recorder.py](./examples/metric_recorder.py)

## Reference

- [Matlab Code](https://github.com/DengPingFan/CODToolbox) by DengPingFan(<https://github.com/DengPingFan>): In our comparison (the test code can be seen under the `test` folder), the result is consistent with the code.
  - The matlab code needs to change `Bi_sal(sal>threshold)=1;` to `Bi_sal(sal>=threshold)=1;` in <https://github.com/DengPingFan/CODToolbox/blob/910358910c7824a4237b0ea689ac9d19d1958d11/Onekey_Evaluation_Code/OnekeyEvaluationCode/main.m#L102>. For related discussion, please see [the issue](https://github.com/DengPingFan/CODToolbox/issues/1).
  - 2021-12-20 (version `1.3.0`): Due to the difference between numpy and matlab, in version `1.2.x`, there are very slight differences on some metrics between the results of the matlab code and ours. The [recent PR](https://github.com/lartpang/PySODMetrics/pull/3) alleviated this problem. However, there are still very small differences on E-measure. The results in most papers are rounded off to three or four significant figures, so, there is no obvious difference between the new version and the version `1.2.x` for them.
- <https://en.wikipedia.org/wiki/Precision_and_recall>

```text
@inproceedings{Fmeasure,
    title={Frequency-tuned salient region detection},
    author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
    booktitle=CVPR,
    number={CONF},
    pages={1597--1604},
    year={2009}
}

@inproceedings{MAE,
    title={Saliency filters: Contrast based filtering for salient region detection},
    author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
    booktitle=CVPR,
    pages={733--740},
    year={2012}
}

@inproceedings{Smeasure,
    title={Structure-measure: A new way to evaluate foreground maps},
    author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
    booktitle=ICCV,
    pages={4548--4557},
    year={2017}
}

@inproceedings{Emeasure,
    title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
    author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
    booktitle=IJCAI,
    pages="698--704",
    year={2018}
}

@inproceedings{wFmeasure,
  title={How to evaluate foreground maps?},
  author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
  booktitle=CVPR,
  pages={248--255},
  year={2014}
}

@inproceedings{MSIoU,
    title = {Multiscale IOU: A Metric for Evaluation of Salient Object Detection with Fine Structures},
    author = {Ahmadzadeh, Azim and Kempton, Dustin J. and Chen, Yang and Angryk, Rafal A.},
    booktitle = ICIP,
    year = {2021},
}
```
