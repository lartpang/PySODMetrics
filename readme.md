# PySODMetrics

![logo](./docs/imgs/logo.png)

[中文介绍](./docs/readme_zh.md)

## Introduction

A simple and efficient implementation of SOD metrcis

- Based on `numpy` and `scipy`
- Verification based on Fan's matlab code <https://github.com/DengPingFan/CODToolbox>
- The code structure is simple and easy to extend
- The code is lightweight and fast

Your improvements and suggestions are welcome.

## TODO List

* [X] Speed up the calculation of Emeasure.
* [ ] Add the necessary documentation for different functions.

## Compared with Matlab Code from Fan <https://github.com/DengPingFan/CODToolbox>

In our comparison (the test code can be seen under the `test` folder), the result is consistent with Fan's code, as follows:

```text
ours:   Smeasure:0.903; wFmeasure:0.558; MAE:0.037; adpEm:0.941; meanEm:0.957; maxEm:0.967; adpFm:0.582; meanFm:0.577; maxFm:0.589
matlab: Smeasure:0.903; wFmeasure:0.558; MAE:0.037; adpEm:0.941; meanEm:0.957; maxEm:0.967; adpFm:0.582; meanFm:0.577; maxFm:0.589.
```

**NOTE** 

The matlab code based here <https://github.com/DengPingFan/CODToolbox/blob/910358910c7824a4237b0ea689ac9d19d1958d11/Onekey_Evaluation_Code/OnekeyEvaluationCode/main.m#L102> 
needs to change `Bi_sal(sal>threshold)=1;` to `Bi_sal(sal>=threshold)=1;`. 

For related discussion, please see: <https://github.com/DengPingFan/CODToolbox/issues/1>

## Usage

### Download the file as your script

```shell script
wget -nc -O metrics.py https://raw.githubusercontent.com/lartpang/PySODMetrics/main/sod_metrics/__init__.py
```

`-nc`: If the file 'metrics.py' already exists, it cannot be retrieved.

### Requirements

```shell
pip install -r requirements.txt
```

### Examples
 
 * <./tests/test_metrics.py>

## Thanks

* <https://github.com/DengPingFan/CODToolbox> 
    - By DengPingFan(<https://github.com/DengPingFan>)

## Reference

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
