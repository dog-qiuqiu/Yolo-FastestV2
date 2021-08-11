# Yolo-FastestV2
# Evaluating indicator/Benchmark
Network|COCO mAP(0.5)|Resolution|Run Time(Ncnn 4xCore)|Run Time(Ncnn 1xCore)|FLOPs(G)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Yolo-FastetsV2|23.56 %|352X352|3.23 ms|4.5 ms|0.238|0.25M
[Yolo-Fastest-1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40 %|320X320|5.59 ms|7.52 ms|0.252|0.35M
[Yolov4-Tiny](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)|40.2%|416X416|23.67ms|40.14ms|6.9|5.77M

* ***Test platform Mi 11 Snapdragon 888 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* COCO 2017 Val mAP（no group label）
* Suitable for hardware with extremely tight computing resources
