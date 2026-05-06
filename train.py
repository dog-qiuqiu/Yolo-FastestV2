import os
import math
import time
import random
import argparse

import numpy as np
import torch
from torch import optim
from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
import model.detector as detector_mod   # 避免与下面的 net 实例同名


def parse_args():
    parser = argparse.ArgumentParser(description="Yolo-FastestV2 训练")
    parser.add_argument('--data', type=str, required=True,
                        help='训练配置 *.data 路径')
    parser.add_argument('--keep-ratio', dest='keep_ratio', action='store_true',
                        default=False, help='预处理使用 letterbox 等比缩放 (默认关, 可选)')
    parser.add_argument('--no-keep-ratio', dest='keep_ratio', action='store_false',
                        help='预处理直接 stretch 拉伸 (默认, 与原仓库行为一致)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子, 便于复现 (默认 42)')
    parser.add_argument('--device', type=str, default='auto',
                        help="设备: 'auto'(默认) | 'cuda' | 'cuda:0' | 'cpu'")
    parser.add_argument('--resume', type=str, default='',
                        help='从指定 .pth 续训 (留空则按 cfg.pre_weights / 默认 backbone 初始化)')
    parser.add_argument('--save-dir', type=str, default='weights',
                        help='权重保存目录')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(spec: str):
    if spec == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(spec)


def main():
    opt = parse_args()
    set_seed(opt.seed)

    cfg = utils.utils.load_datafile(opt.data)
    print("训练配置:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print(f"  keep_ratio: {opt.keep_ratio}")
    print(f"  seed: {opt.seed}")

    # 数据集
    train_dataset = utils.datasets.TensorDataset(
        cfg["train"], cfg["width"], cfg["height"],
        imgaug=True, keep_ratio=opt.keep_ratio,
    )
    val_dataset = utils.datasets.TensorDataset(
        cfg["val"], cfg["width"], cfg["height"],
        imgaug=False, keep_ratio=opt.keep_ratio,
    )

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    persistent = nw > 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=utils.datasets.collate_fn, num_workers=nw,
        pin_memory=True, drop_last=True, persistent_workers=persistent,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=utils.datasets.collate_fn, num_workers=nw,
        pin_memory=True, drop_last=False, persistent_workers=persistent,
    )

    # 设备
    device = select_device(opt.device)
    print(f"  device: {device}")

    # 模型
    # cfg["pre_weights"] 是字符串 "None" / 文件路径 / 空, 这里按字符串"None"安全处理
    pre_weights = cfg.get("pre_weights")
    use_backbone_init = not (pre_weights and pre_weights != "None"
                             and os.path.exists(pre_weights))
    net = detector_mod.Detector(cfg["classes"], cfg["anchor_num"], use_backbone_init).to(device)
    summary(net, input_size=(3, cfg["height"], cfg["width"]))

    start_epoch = 0
    if opt.resume:
        assert os.path.exists(opt.resume), f"找不到 resume 权重: {opt.resume}"
        ckpt = torch.load(opt.resume, map_location=device)
        # 支持 dict {'epoch','state_dict','optimizer'} 或纯 state_dict
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            net.load_state_dict(ckpt['state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"[resume] 从 {opt.resume} 续训, 起始 epoch={start_epoch}")
        else:
            net.load_state_dict(ckpt)
            print(f"[resume] 加载权重 {opt.resume}, 但无 epoch 信息, 从 0 开始")
    elif pre_weights and pre_weights != "None" and os.path.exists(pre_weights):
        net.load_state_dict(torch.load(pre_weights, map_location=device), strict=False)
        print(f"[finetune] 加载预训练权重: {pre_weights}")
    else:
        print("[init] 使用 model/backbone/backbone.pth 初始化")

    # 优化器
    optimizer = optim.SGD(
        params=net.parameters(),
        lr=cfg["learning_rate"],
        momentum=0.949,
        weight_decay=0.0005,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["steps"], gamma=0.1,
    )

    # 续训时把 scheduler 拨到对应位置
    for _ in range(start_epoch):
        scheduler.step()

    os.makedirs(opt.save_dir, exist_ok=True)
    last_path = os.path.join(opt.save_dir, f"{cfg['model_name']}-last.pth")
    best_path = os.path.join(opt.save_dir, f"{cfg['model_name']}-best.pth")

    best_ap = -1.0
    print(f'Starting training for {cfg["epochs"]} epochs (start={start_epoch})...')

    batch_num = start_epoch * len(train_dataloader)
    for epoch in range(start_epoch, cfg["epochs"]):
        net.train()

        epoch_t0 = time.perf_counter()
        for imgs, targets in train_dataloader:
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            preds = net(imgs)
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(
                preds, targets, cfg, device
            )
            total_loss.backward()

            # 学习率预热
            warmup_num = 5 * len(train_dataloader)
            for g in optimizer.param_groups:
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num / max(warmup_num, 1), 4)
                    g['lr'] = cfg["learning_rate"] * scale
                lr = g["lr"]

            if batch_num % cfg["subdivisions"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            batch_num += 1

        epoch_dt = time.perf_counter() - epoch_t0

        # epoch 末打印 (避免被 tqdm 进度条刷屏)
        print(
            f"Epoch:{epoch:3d} LR:{lr:.6f} "
            f"CIou:{float(iou_loss):.4f} Obj:{float(obj_loss):.4f} "
            f"Cls:{float(cls_loss):.4f} Total:{float(total_loss):.4f} "
            f"({epoch_dt:.1f}s)"
        )

        # 保存 last 权重 (每 epoch 都覆盖一次, 便于中断恢复)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
        }, last_path)

        # 评估并维护 best
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg["epochs"]:
            net.eval()
            print("computer mAP...")
            _, _, AP, _ = utils.utils.evaluation(val_dataloader, cfg, net, device)
            print("computer PR...")
            precision, recall, _, f1 = utils.utils.evaluation(
                val_dataloader, cfg, net, device, 0.3
            )
            print(f"[eval] Epoch:{epoch} P:{precision:.4f} R:{recall:.4f} "
                  f"AP:{AP:.4f} F1:{f1:.4f}")

            if AP > best_ap:
                best_ap = AP
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'mAP': AP,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }, best_path)
                print(f"[best] mAP 提升到 {AP:.4f}, 保存 -> {best_path}")

        scheduler.step()

    print(f"\n训练结束. 最佳 mAP = {best_ap:.4f}, 权重: {best_path}")


if __name__ == '__main__':
    main()
