# 数据来源

* `aerial-1867541__340` SOC 纯背景mask
* `0001.png` ECSSD 包含目标的mask
* `19.png` Pascal-S
* `imgs/*.png` 由 `skimage.data.astronaut()`（NASA，公有领域）通过 OpenCV 4.13 的 `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` 缩放到对应 mask 的尺寸后保存（RGB 图像以 BGR 顺序写入 PNG），作为 `CamouflageContextMeasure` 的输入图像。由于 OpenCV 4.14 起 `cv2.resize` 的输出发生了变化，这里直接保存缩放后的结果，以保证测试结果不受 OpenCV 版本影响。
