import os
import argparse
import model.detector
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 这一行不写后面的数据拷贝会出错

import utils.utils
import utils.datasets
import json
import numpy as np
import torch
import numpy as np
import struct

# int8量化参考代码 https://blog.csdn.net/oYeZhou/article/details/106719154
def unpack_data(binpath):
    nums = int(os.path.getsize(binpath)/4)
    with open(binpath, 'rb') as f:
        data = struct.unpack('f'*nums, f.read(4*nums))
    data = torch.tensor(data)
    return data

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', 
                        help='Specify config')
    parser.add_argument('--weights', '-w', type=str, default='', 
                        help='The path of the .trt model to be transformed')
    parser.add_argument('--testdir','-t', type=str, default='', 
                        help='The path of test data')

    args = parser.parse_args()
    assert os.path.exists(args.weights), "请指定正确的模型路径"

    BATCH_SIZE = 1
    # 2. 选择是否采用FP16精度还是int8或者float32，与导出的trt模型保持一致                                  
    target_dtype = np.float32  

    # 3. 创建Runtime，加载TRT引擎
    f = open(args.weights, "rb")                            # 读取trt模型
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))   # 创建一个Runtime(传入记录器Logger)
    engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
    context = engine.create_execution_context()             # 创建context
    f.close()

    # 4. 分配input和output内存
    input_batch = np.random.randn(BATCH_SIZE, 80448).astype(target_dtype)
    output0 = np.empty([BATCH_SIZE, 12, 22, 15], dtype = target_dtype)
    output1 = np.empty([BATCH_SIZE, 6, 11, 15], dtype = target_dtype)

    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output0 = cuda.mem_alloc(1 * output0.nbytes)
    d_output1 = cuda.mem_alloc(1 * output1.nbytes)

    bindings = [int(d_input), int(d_output0), int(d_output1)]
    stream = cuda.Stream()

    # 5. 创建predict函数
    def predict(batch): # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
        # transfer predictions back
        cuda.memcpy_dtoh_async(output0, d_output0, stream)
        cuda.memcpy_dtoh_async(output1, d_output1, stream)
        # syncronize threads
        stream.synchronize()

        return output0, output1

    with open(args.config) as f:
        cfg = json.load(f)
    cqt_transform = model.detector.CQTSpectrogram(cfg["cqt"], cfg["m_config"]["width"], cfg["m_config"]["height"], interpolate=False)

    batch_size = int(cfg["opt"]["batch_size"] / cfg["opt"]["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    batch_size = 1
    # filepaths = [os.path.join(args.testdir, file) for file in os.listdir(args.testdir)]
    filepaths = ["/home/data/SSVD-v2.0/test16k/100135.wav"]

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fileno, filepath in enumerate(filepaths):
        cfg["cqt"]["overlap_ratio"] = cfg["test"]["overlap_ratio"]
        testset = utils.datasets.TestDataset(cfg["cqt"], filepath)
        loader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             drop_last=False,
                                             persistent_workers=True
                                             )
        output_boxes = []
        for index, data in enumerate(loader): # [6, 80448] 等效于多张频谱图
            data1 = data.numpy()
            # data2 = unpack_data('/home/data/wxk/Yolo-FastestV2/sample/tensorrt/build/cqt.bin').numpy()
            data = data1

            preds = predict(data)
            preds = [torch.from_numpy(x) for x in preds]

            # 把预测出来的归一化坐标转换到yolox输入图像的尺寸
            output = utils.utils.handel_preds_trt(preds, cfg)
            # 可以当做一个二维list，第一层代表第几个数据，第二层为每张图像上框的数量
            output_box = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
            output_boxes.extend(output_box)

        scale_h = cfg["cqt"]["n_bins"] / (cfg["cqt"]["bins_per_octave"] / 12) / cfg["m_config"]["height"]
        scale_w = cfg["cqt"]["duration"] / cfg["m_config"]["width"]
        hop = (1. - cfg["cqt"]["overlap_ratio"]) * cfg["m_config"]["width"]
        total_notes = utils.utils.convert_boxs_to_notes(output_boxes, hop, scale_h, scale_w, cfg["m_config"]["width"])
        total_notes.sort(key=lambda x: x[0])
        feature = cqt_transform(testset.get_total_audio())[0, 0].numpy()

        if cfg["test"]["test_checkdata_dir"]:
            os.makedirs(cfg["test"]["test_checkdata_dir"], exist_ok=True)
            dumppath = os.path.join(cfg["test"]["test_checkdata_dir"], os.path.basename(filepath).split('.')[0] + '.png')
            utils.utils.dump_test_data(feature, dumppath, total_notes, cfg["cqt"])
            exit(0)