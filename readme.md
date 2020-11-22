# 一个简单直接的SOD指标工具箱

- 基于`numpy`和极少量`scipy.ndimage`代码
- 基于DengPing Fan <https://github.com/DengPingFan/CODToolbox>
- 结构简单，易于扩展
- 代码轻量且快速

欢迎您的改进和建议。

## 已知问题

在我们的测试中 (测试代码可见`test`文件夹下内容)，大部分结果与Fan的代码一致，如下:

```text
ours:   Smeasure 0.959  wFmeasure 0.438  MAE 0.018  adpEm 0.946  meanEm 0.975  maxEm 0.987  adpFm 0.456  meanFm 0.454  maxFm 0.461
matlab: Smeasure:0.959; wFmeasure:0.438; MAE:0.018; adpEm:0.946; meanEm:0.977; maxEm:0.987; adpFm:0.456; meanFm:0.454; maxFm:0.461.
```
其中在`meanEm`上会有些许误差，暂未搞明白具体原因所在，但是并不影响最终的使用。

欢迎您指出问题。

## Thanks

* <https://github.com/DengPingFan/CODToolbox> 
    - By DengPingFan(<https://github.com/DengPingFan>)

## 相关文献

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
