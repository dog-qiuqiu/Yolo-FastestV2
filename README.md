# :zap:Yolo-FastestV2:zap:
![image](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/img/demo.png)
* Simple, fast, compact, easy to transplant
* Less resource occupation, excellent single-core performance, lower power consumption
* Faster and smaller:Trade 1% loss of accuracy for 40% increase in inference speed, reducing the amount of parameters by 25%
* Fast training speed, low computing power requirements, training only requires 3GB video memory, gtx1660ti training COCO 1 epoch only takes 7 minutes
# Evaluating indicator/Benchmark
Network|COCO mAP(0.5)|Resolution|Run Time(4xCore)|Run Time(1xCore)|FLOPs(G)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2/tree/main/modelzoo)|23.56 %|352X352|3.23 ms|4.5 ms|0.238|0.25M
[Yolo-FastestV1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40 %|320X320|5.59 ms|7.52 ms|0.252|0.35M
[Yolov4-Tiny](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)|40.2%|416X416|23.67ms|40.14ms|6.9|5.77M

* ***Test platform Mi 11 Snapdragon 888 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
* Reasons for the increase in inference speed: optimization of model memory access
* Suitable for hardware with extremely tight computing resources
# How to use
## Dependent installation
  * PIP
  ```
  pip3 install -r requirements.txt
  ```
## Test
* Picture test
  ```
  python3 test.py --data data/coco.data --weights modelzoo/coco2017-epoch-0.235624ap-model.pth --img img/dog.jpg
  ```
![image](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/img/dog_result.png)
## How to train
### Building data sets(The dataset is constructed in the same way as darknet yolo)
* The format of the data set is the same as that of Darknet Yolo, Each image corresponds to a .txt label file. The label format is also based on Darknet Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  .
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path .txt file, the example content is as follows：
  
  train.txt
  ```
  /home/qiuqiu/Desktop/dataset/train/000001.jpg
  /home/qiuqiu/Desktop/dataset/train/000002.jpg
  /home/qiuqiu/Desktop/dataset/train/000003.jpg
  ```
  val.txt
  ```
  /home/qiuqiu/Desktop/dataset/val/000070.jpg
  /home/qiuqiu/Desktop/dataset/val/000043.jpg
  /home/qiuqiu/Desktop/dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:
 
  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  .
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Get anchor bias
* Generate anchor based on current dataset
  ```
  python3 genanchors.py --traintxt ./train.txt
  ```
* The anchors6.txt file will be generated in the current directory,the sample content of the anchors6.txt is as follows:
  ```
  12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87  # anchor bias
  0.636158                                                                             # iou
  ```
### Build the training .data configuration file
* Reference./data/coco.data
  ```
  [name]
  model_name=coco           # model name

  [train-configure]
  epochs=300                # train epichs
  steps=150,250             # Declining learning rate steps
  batch_size=64             # batch size
  subdivisions=1            # Same as the subdivisions of the darknet cfg file
  learning_rate=0.001       # learning rate

  [model-configure]
  pre_weights=None          # The path to load the model, if it is none, then restart the training
  classes=80                # Number of detection categories
  width=352                 # The width of the model input image
  height=352                # The height of the model input image
  anchor_num=3              # anchor num
  anchors=12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87 #anchor bias

  [data-configure]
  train=/media/qiuqiu/D/coco/train2017.txt   # train dataset path .txt file
  val=/media/qiuqiu/D/coco/val2017.txt       # val dataset path .txt file 
  names=./data/coco.names                    # .names category label file
  ```
### Train
* Perform training tasks
  ```
  python3 train.py --data data/coco.data
  ```
### Evaluation
* Calculate map evaluation
  ```
  python3 evaluation.py --data data/coco.data --weights modelzoo/coco2017-epoch-0.235624ap-model.pth
  ```
# Deploy
## NCNN
