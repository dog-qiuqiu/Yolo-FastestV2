# :zap:Yolo-FastestV2:zap:
![image](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/img/demo.png)
* Simple, fast, compact, easy to transplant
* Less resource occupation, excellent single-core performance, lower power consumption
* Faster and smaller:Trade 1% loss of accuracy for 40% increase in inference speed, reducing the amount of parameters by 25%
# Evaluating indicator/Benchmark
Network|COCO mAP(0.5)|Resolution|Run Time(4xCore)|Run Time(1xCore)|FLOPs(G)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Yolo-FastestV2|23.56 %|352X352|3.23 ms|4.5 ms|0.238|0.25M
[Yolo-FastestV1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40 %|320X320|5.59 ms|7.52 ms|0.252|0.35M
[Yolov4-Tiny](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)|40.2%|416X416|23.67ms|40.14ms|6.9|5.77M

* ***Test platform Mi 11 Snapdragon 888 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* Suitable for hardware with extremely tight computing resources
