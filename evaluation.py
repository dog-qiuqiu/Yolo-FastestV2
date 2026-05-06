import os
import argparse

import torch
from torchsummary import summary

import utils.utils
import utils.datasets
import model.detector as detector_mod


def parse_args():
    parser = argparse.ArgumentParser(description="Yolo-FastestV2 验证集评估")
    parser.add_argument('--data', type=str, required=True,
                        help='训练配置 *.data 路径')
    parser.add_argument('--weights', type=str, required=True,
                        help='待评估的 .pth 权重路径')
    parser.add_argument('--keep-ratio', dest='keep_ratio', action='store_true',
                        default=False, help='预处理使用 letterbox 等比缩放 (默认关)')
    parser.add_argument('--no-keep-ratio', dest='keep_ratio', action='store_false',
                        help='预处理直接 stretch 拉伸 (默认)')
    parser.add_argument('--conf', type=float, default=0.01,
                        help='评估时使用的置信度阈值 (mAP 计算默认 0.01, PR 默认 0.3)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='匹配 GT 用的 IoU 阈值')
    parser.add_argument('--device', type=str, default='auto',
                        help="设备: 'auto'(默认) | 'cuda' | 'cuda:0' | 'cpu'")
    return parser.parse_args()


def main():
    opt = parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"

    print("评估配置:")
    print(f"  model_name : {cfg['model_name']}")
    print(f"  width x H  : {cfg['width']} x {cfg['height']}")
    print(f"  val list   : {cfg['val']}")
    print(f"  weights    : {opt.weights}")
    print(f"  keep_ratio : {opt.keep_ratio}")
    print(f"  conf / iou : {opt.conf} / {opt.iou}")

    val_dataset = utils.datasets.TensorDataset(
        cfg["val"], cfg["width"], cfg["height"],
        imgaug=False, keep_ratio=opt.keep_ratio,
    )

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    persistent = nw > 0
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=utils.datasets.collate_fn, num_workers=nw,
        pin_memory=True, drop_last=False, persistent_workers=persistent,
    )

    # 设备
    if opt.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)

    net = detector_mod.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    try:
        state = torch.load(opt.weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(opt.weights, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    net.load_state_dict(state)
    net.eval()

    summary(net, input_size=(3, cfg["height"], cfg["width"]))

    print("computer mAP...")
    _, _, AP, _ = utils.utils.evaluation(
        val_dataloader, cfg, net, device,
        conf_thres=opt.conf, iou_thres=opt.iou,
    )
    print("computer PR...")
    precision, recall, _, f1 = utils.utils.evaluation(
        val_dataloader, cfg, net, device,
        conf_thres=0.3, iou_thres=opt.iou,
    )
    print(f"Precision:{precision:.4f} Recall:{recall:.4f} AP:{AP:.4f} F1:{f1:.4f}")


if __name__ == '__main__':
    main()
