"""
单张图推理脚本.
默认 letterbox 等比缩放, 与训练/验证保持一致.
"""
import os
import time
import argparse

import cv2
import numpy as np
import torch

import utils.utils
import utils.datasets
import model.detector as detector_mod


def parse_args():
    parser = argparse.ArgumentParser(description="Yolo-FastestV2 单图推理")
    parser.add_argument('--data', type=str, required=True,
                        help='训练配置 *.data 路径')
    parser.add_argument('--weights', type=str, required=True,
                        help='.pth 权重路径')
    parser.add_argument('--img', type=str, required=True,
                        help='测试图像路径')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.4,
                        help='NMS IoU 阈值')
    parser.add_argument('--keep-ratio', dest='keep_ratio', action='store_true',
                        default=False, help='预处理使用 letterbox (默认关)')
    parser.add_argument('--no-keep-ratio', dest='keep_ratio', action='store_false',
                        help='预处理直接 stretch (默认)')
    parser.add_argument('--out', type=str, default='test_result.png',
                        help='输出文件名')
    parser.add_argument('--device', type=str, default='auto',
                        help="设备: 'auto'(默认) | 'cuda' | 'cuda:0' | 'cpu'")
    return parser.parse_args()


def main():
    opt = parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    # 设备
    if opt.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)

    # 模型
    net = detector_mod.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    try:
        state = torch.load(opt.weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(opt.weights, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    net.load_state_dict(state)
    net.eval()

    # 数据预处理 (与训练对齐)
    ori_img = cv2.imread(opt.img)
    ori_h, ori_w = ori_img.shape[:2]

    if opt.keep_ratio:
        canvas, ratio, pad_l, pad_t = utils.datasets.letterbox(
            ori_img, cfg["width"], cfg["height"]
        )
    else:
        canvas = cv2.resize(ori_img, (cfg["width"], cfg["height"]),
                            interpolation=cv2.INTER_LINEAR)
        ratio, pad_l, pad_t = None, 0, 0

    img = canvas.transpose(2, 0, 1)[None]
    img = torch.from_numpy(img).to(device).float() / 255.0

    # 模型推理 (含 CUDA sync 才能测准时间)
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = net(img)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    forward_ms = (t1 - t0) * 1000.0
    print(f"forward time: {forward_ms:.2f} ms")

    # 后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(
        output, conf_thres=opt.conf, iou_thres=opt.iou
    )

    # label 名称
    with open(cfg["names"], 'r', encoding='utf-8') as f:
        LABEL_NAMES = [line.strip() for line in f if line.strip()]

    # 反变换到原图坐标
    if opt.keep_ratio:
        def to_orig(b):
            x1 = (b[0] - pad_l) / ratio
            y1 = (b[1] - pad_t) / ratio
            x2 = (b[2] - pad_l) / ratio
            y2 = (b[3] - pad_t) / ratio
            return x1, y1, x2, y2
    else:
        sw, sh = ori_w / cfg["width"], ori_h / cfg["height"]
        def to_orig(b):
            return b[0] * sw, b[1] * sh, b[2] * sw, b[3] * sh

    for box in output_boxes[0]:
        b = box.tolist()
        x1, y1, x2, y2 = to_orig(b)
        x1 = int(max(0, min(ori_w - 1, x1)))
        y1 = int(max(0, min(ori_h - 1, y1)))
        x2 = int(max(0, min(ori_w - 1, x2)))
        y2 = int(max(0, min(ori_h - 1, y2)))
        score = b[4]
        cls_id = int(b[5])
        category = LABEL_NAMES[cls_id] if 0 <= cls_id < len(LABEL_NAMES) else str(cls_id)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(opt.out, ori_img)
    print(f"已保存 -> {opt.out}")


if __name__ == '__main__':
    main()
