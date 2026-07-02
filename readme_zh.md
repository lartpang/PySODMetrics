<div align="center">
  <img src="https://github.com/lartpang/PySODMetrics/blob/main/images/logo.png?raw=true" alt="Logo" width="320" height="auto">
  </br>
  <h2>PySODMetrics: 一份简单有效的 SOD 指标实现</h2>
  <a href="./readme.md"><img src="https://img.shields.io/badge/README-English-blue"></a>
  <a href="https://arxiv.org/abs/2607.00886"><img src="https://img.shields.io/badge/arXiv-2607.00886-b31b1b.svg"></a>
  <img src="https://img.shields.io/pypi/v/pysodmetrics">
  <img src="https://img.shields.io/pypi/dm/pysodmetrics?label=pypi%20downloads&logo=PyPI&logoColor=white">
  <img src="https://img.shields.io/github/last-commit/lartpang/PySODMetrics">
  <img src="https://img.shields.io/github/release-date/lartpang/PySODMetrics">
</div>

> [!important]
>
> 我们在这一领域的探索仍在继续，这一次是通过 [PyIRSTDMetrics](https://github.com/lartpang/PyIRSTDMetrics) —— 一个源于同样核心动机的项目。
> 你可以把它们看作是一对“孪生”计划：这个项目旨在描绘当前评估的全景，而它的“兄弟”项目则更进一步，在此基础上加以拓展，并重新思考这一体系。
> 我们很希望得到你的 Star！ 🌟


## 介绍

一份简单有效的 SOD 指标实现。

## 引用

如果你在研究中使用了本库，请考虑引用以下论文：

```bibtex
@misc{pang2026pixeloverlapframeworkdecomposing,
      title={Beyond Pixel Overlap: A Framework for Decomposing Segmentation Evaluation Metrics}, 
      author={Youwei Pang and Xiaoqi Zhao},
      year={2026},
      eprint={2607.00886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.00886}, 
}
```

- 基于 `numpy` 和极少量 `scipy.ndimage` 代码
- 基于 Deng-Ping Fan 的 Matlab 代码 <https://github.com/DengPingFan/CODToolbox> 进行对比验证
- 结构简单，易于扩展
- 代码轻量且快速

欢迎您的改进和建议。

## 相关项目

- [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit): 一个面向显著目标检测和伪装目标检测的 Python 评估工具箱

### 支持的指标

| 指标                                                | Sample-based                                | Whole-based              | 相关类                                       |
| --------------------------------------------------- | ------------------------------------------- | ------------------------ | -------------------------------------------- |
| MAE                                                 | soft,si-soft                                |                          | `MAE`                                        |
| S-measure $S_{m}$                                   | soft                                        |                          | `Smeasure`                                   |
| weighted F-measure ($F^{\omega}_{\beta}$)          | soft                                        |                          | `WeightedFmeasure`                           |
| Human Correction Effort Measure                     | soft                                        |                          | `HumanCorrectionEffortMeasure`               |
| Context-Measure ($C_{\beta}$, $C^{\omega}_{\beta}$) | soft                                        |                          | `ContextMeasure`, `CamouflageContextMeasure` |
| Multi-Scale IoU                                     | max,avg,adp,bin                             |                          | `MSIoU`                                      |
| E-measure ($E_{m}$)                                 | max,avg,adp                                 |                          | `Emeasure`                                   |
| F-measure (old) ($F_{\beta}$)                       | max,avg,adp                                 |                          | `Fmeasure`（后续将移除）                     |
| F-measure (new) ($F_{\beta}$, $F_{1}$)              | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`FmeasureHandler`               |
| BER                                                 | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`BERHandler`                    |
| Dice                                                | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`DICEHandler`                   |
| FPR                                                 | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`FPRHandler`                    |
| IoU                                                 | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`IOUHandler`                    |
| Kappa                                               | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`KappaHandler`                  |
| Overall Accuracy                                    | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`OverallAccuracyHandler`        |
| Precision                                           | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`PrecisionHandler`              |
| Recall                                              | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`RecallHandler`                 |
| Sensitivity                                         | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`SensitivityHandler`            |
| Specificity                                         | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`SpecificityHandler`            |
| TNR                                                 | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`TNRHandler`                    |
| TPR                                                 | max,avg,adp,bin,si-max,si-avg,si-adp,si-bin | bin,si-max,si-avg,si-bin | `FmeasureV2`+`TPRHandler`                    |

**注意**：
- 基于样本（Sample-based）的 `si-` 变体会根据形状为 `(num_targets, 256)` 的样本级 `fm` 序列，计算每个样本的均值/最大值。
- 基于整体（Whole-based）的 `si-` 变体会在跨所有样本的所有目标的平均 `fm` 序列上计算均值/最大值。
- 由于 `*adp` 变体是专门为 `sample-based` 计算设计的，因此不支持基于整体（whole-based）的计算方式。

## 使用

核心文件在文件夹 `py_sod_metrics` 中。

- **[最新，但可能不稳定]** 从源代码安装：`pip install git+https://github.com/lartpang/PySODMetrics.git`
- **[更稳定]** 从 PyPI 安装：`pip install pysodmetrics`

### 示例

- [examples/test_metrics.py](./examples/test_metrics.py)
- [examples/metric_recorder.py](./examples/metric_recorder.py)

## 参考

- [Matlab Code](https://github.com/DengPingFan/CODToolbox) by DengPingFan(<https://github.com/DengPingFan>)：在我们的测试中（测试代码可见 `test` 文件夹下内容），结果与 Fan 的代码一致。
  - Matlab 代码需要将 <https://github.com/DengPingFan/CODToolbox/blob/910358910c7824a4237b0ea689ac9d19d1958d11/Onekey_Evaluation_Code/OnekeyEvaluationCode/main.m#L102> 中的 `Bi_sal(sal>threshold)=1;` 改为 `Bi_sal(sal>=threshold)=1;`。细节可见 [相关讨论](https://github.com/DengPingFan/CODToolbox/issues/1)。
  - 2021-12-20（version `1.3.0`）：由于 numpy 和 Matlab 的差异，在 `1.2.x` 版本中，Matlab 代码的结果与我们的结果在某些指标上存在非常细微的差异。[最近的 PR](https://github.com/lartpang/PySODMetrics/pull/3) 缓解了这个问题。但是，在 E-measure 上仍然存在非常小的差异。大多数论文中的结果都会四舍五入到三四位有效数字，因此，新版本与 `1.2.x` 版本之间没有明显差异。
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
@inproceedings{SizeInvarianceVariants,
    title = {Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection},
    author = {Feiran Li and Qianqian Xu and Shilong Bao and Zhiyong Yang and Runmin Cong and Xiaochun Cao and Qingming Huang},
    booktitle = ICML,
    year = {2024}
}
@inproceedings{HumanCorrectionEffortMeasure,
    title = {Highly Accurate Dichotomous Image Segmentation},
    author = {Xuebin Qin and Hang Dai and Xiaobin Hu and Deng-Ping Fan and Ling Shao and Luc Van Gool},
    booktitle = ECCV,
    year = {2022}
}
@article{ContextMeasure,
    title={Context-measure: Contextualizing Metric for Camouflage},
    author={Wang, Chen-Yang and Ji, Gepeng and Shao, Song and Cheng, Ming-Ming and Fan, Deng-Ping},
    journal={arXiv preprint arXiv:2512.07076},
    year={2025}
}
```
