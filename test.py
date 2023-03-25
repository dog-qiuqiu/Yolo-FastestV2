import os
import argparse

import torch
import model.detector
import utils.utils
import utils.datasets
import json
import numpy as np

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='', 
                        help='Specify config')
    parser.add_argument('--weights', '-w', type=str, default='', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--testdir','-t', type=str, default='', 
                        help='The path of test data')

    args = parser.parse_args()
    assert os.path.exists(args.weights), "请指定正确的模型路径"

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = json.load(f)

    model = model.detector.Detector(
        cfg["cqt"], cfg["m_config"]["width"], cfg["m_config"]["height"], 
        cfg["m_config"]["anchor_num"], False,
        convert2image=cfg["m_config"]["convert2image"]).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    #sets the module in eval node
    model.eval()
    batch_size = int(cfg["opt"]["batch_size"] / cfg["opt"]["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    batch_size = 128
    h = cfg["cqt"]["n_bins"]
    w = int(cfg["cqt"]["duration"] * cfg["cqt"]["sr"] / cfg["cqt"]["hop"])

    scale_h, scale_w = h / cfg["m_config"]["height"], w / cfg["m_config"]["width"]
    filepaths = [os.path.join(args.testdir, file) for file in os.listdir(args.testdir)][1: ]

    for filepath in filepaths:
        print(filepath)
        cfg["cqt"]["overlap_ratio"] = cfg["test"]["overlap_ratio"]
        testset = utils.datasets.TestDataset(cfg["cqt"], filepath)
        loader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             drop_last=False,
                                             persistent_workers=True
                                             )
        for index, data in enumerate(loader): # [6, 80448] 等效于多张频谱图
            data = data.to(device)
            preds = model(data)
            cqt = preds[-1]
            preds = preds[: -1]

            # 把预测出来的归一化坐标转换到yolox输入图像的尺寸
            output = utils.utils.handel_preds(preds, cfg, device)
            # 可以当做一个二维list，第一层代表第几个数据，第二层为每张图像上框的数量
            output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
            total_notes = []
            total_boxs = []
            confs = []
            for feature_idx, output_box in enumerate(output_boxes):
                notes = []
                for box in output_box:
                    box = box.tolist()
                    x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                    x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
                    onset = x1 * cfg["cqt"]["hop"] / cfg["cqt"]["sr"]
                    offset = x2 * cfg["cqt"]["hop"] / cfg["cqt"]["sr"]
                    pitch = y1 / (cfg["cqt"]["bins_per_octave"] / 12) + 21
                    notes.append([onset, offset, pitch])
                    x1 = box[0] / cfg["m_config"]["width"]
                    y1 = box[1] / cfg["m_config"]["height"]
                    x2 = box[2] / cfg["m_config"]["width"]
                    y2 = box[3] / cfg["m_config"]["height"]
                    w, h = x2 - x1, y2 - y1
                    x0 = x1 + w / 2
                    y0 = y1 + h / 2
                    total_boxs.append([feature_idx, x0, y0, w, h])
                    confs.append(box[-1])

                notes.sort(key=lambda x: x[0])
                total_notes.append(notes)

            if cfg["test"]["test_checkdata_dir"]:
                utils.utils.dump_data(cqt.cpu().numpy(), cfg["test"]["test_checkdata_dir"],np.array(total_boxs), 
                                      confs, rgb=cfg["m_config"]["convert2image"])
                exit(0)