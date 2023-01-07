<div align="center">
  <img src="https://github.com/lartpang/PySODMetrics/blob/main/images/logo.png?raw=true" alt="Logo" width="320" height="auto">
  </br>
  <h2>PySODMetrics: 一份简单有效的SOD指标实现</h2>
  <a href="./readme.md"><img src="https://img.shields.io/badge/README-English-blue"></a>
  <img src="https://img.shields.io/pypi/v/pysodmetrics">
  <img src="https://img.shields.io/pypi/dm/pysodmetrics?label=pypi%20downloads&logo=PyPI&logoColor=white">
  <img src="https://img.shields.io/github/last-commit/lartpang/PySODMetrics">
  <img src="https://img.shields.io/github/last-commit/lartpang/PySODMetrics">
  <img src="https://img.shields.io/github/release-date/lartpang/PySODMetrics">
</div>

## 介绍

一份简单有效的 SOD 指标实现

- 基于`numpy`和极少量`scipy.ndimage`代码
- 基于 DengPing Fan <https://github.com/DengPingFan/CODToolbox>
- 结构简单，易于扩展
- 代码轻量且快速

欢迎您的改进和建议。

### 相关项目

- A Python-based salient object detection and video object segmentation evaluation toolbox. <https://github.com/lartpang/Py-SOD-VOS-EvalToolkit>

## 与[范的 Matlab 代码](https://github.com/DengPingFan/CODToolbox)的比较

在我们的测试中 (测试代码可见`test`文件夹下内容)，结果与 Fan 的代码一致，如下:

```text
ours:   Smeasure:0.903; wFmeasure:0.558; MAE:0.037; adpEm:0.941; meanEm:0.957; maxEm:0.967; adpFm:0.582; meanFm:0.577; maxFm:0.589
matlab: Smeasure:0.903; wFmeasure:0.558; MAE:0.037; adpEm:0.941; meanEm:0.957; maxEm:0.967; adpFm:0.582; meanFm:0.577; maxFm:0.589.
```

**注意**

- matlab 代码需要将<https://github.com/DengPingFan/CODToolbox/blob/910358910c7824a4237b0ea689ac9d19d1958d11/Onekey_Evaluation_Code/OnekeyEvaluationCode/main.m#L102> 的`Bi_sal(sal>threshold)=1;`改为` Bi_sal(sal>=threshold)=1;`。细节可见 [相关讨论](https://github.com/DengPingFan/CODToolbox/issues/1)。
- 2021-12-20 (Version `1.3.0`)：由于 numpy 和 matlab 的不同，在 `1.2.x` 版本中，matlab 代码的结果与我们的结果在某些指标上存在非常细微的差异。[最近的 PR](https://github.com/lartpang/PySODMetrics/pull/3) 缓解了这个问题。但是，在 E-measure 上仍然存在非常小的差异。大多数论文中的结果都四舍五入到三四位有效数字，因此，新版本与“1.2.x”版本之间没有明显差异。

## 使用

### ~~下载文件为自己的脚本~~

核心文件在文件夹 `py_sod_metrics` 中。

### 安装成一个包

```shell script
pip install pysodmetrics
```

### 示例

- <examples/test_metrics.py>
- <examples/metric_recorder.py>

## 感谢

- <https://github.com/DengPingFan/CODToolbox>
  - By DengPingFan(<https://github.com/DengPingFan>)

## 参考文献

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
```
