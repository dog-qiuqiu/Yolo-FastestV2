# MusicYOLO-tensorrt

MusicYOLO-tensorrt is based on Yolo-FastestV2. The model is trained on SSVD v2.0 dataset. 

# Training 
* ***get the anchor box shape from SSVD v2.0 dataset***
  ```python
  python3 yolo_anchor.py -c yolo.json
  ```
  The anchor boxs shapes will be stored in anchors6.txt.
  Copy values to the "anchors" in yolo.json.
* ***prepare data for training***

  Since audios' sampling rate maybe different. Resample all audios to 16KHz sampling rate and stored in numpy data format.
  ```python
  python3 generate_data.py -c yolo.json
  ```

* ***training***
  ```python
  python3 train.py -c yolo.json
  ```

# Inerence
* ***select the best model***

  select the best model from the train.log or the ckpt files' name. Regard the best model ckpt as BESTMODEL_PATH.

* ***tensorrt python inference***
  ```python
  python3 test_long_trt.py -c yolo.json -w BESTMODEL_PATH -t YOUR_AUDIO_DIR
  ```
  
  **NOTE**: All audios should be resampled to 16KHz first!
* ***tensorrt C++ inference***

  ```bash
  cd sample/tensorrt
  mkdir build && cd build
  cmake ..
  make
  ```
  MusicYOLO Flops: 53.45 MMac (5 second audio chunk) Params: 236.82 k. MusicYOLO-tensorrt C++ version runs on a machine with a four cores CPU, a 3060 GPU. an audio of 66.3s occupies one CPU core, CPU memory 1.5G, GPU memory 887M and GPU Util 38%. The total inference time (donnot contain read and write file) is 330ms. 