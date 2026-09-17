import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
from numpy.testing._private.utils import print_assert_equal

import torch
from torch import optim
from torch.utils.data import dataset
from numpy.core.fromnumeric import shape

from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
from utils.coco_evaluation import evaluate_coco
import model.detector


def load_compatible_pretrained_weights(model, checkpoint_path, device):
    """Load checkpoint tensors whose names and shapes match the current model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    compatible_state = {
        name: tensor
        for name, tensor in checkpoint.items()
        if name in model_state and tensor.shape == model_state[name].shape
    }
    skipped_keys = sorted(set(checkpoint) - set(compatible_state))

    model.load_state_dict(compatible_state, strict=False)
    if skipped_keys:
        print("Skip incompatible pretrained parameters: %s" % ", ".join(skipped_keys))


if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.data',
                        help='Specify training profile *.data')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader worker processes (use 0 if multiprocessing is unavailable)')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    print("训练配置:")
    print(cfg)

    # 数据集加载
    train_dataset = utils.datasets.TensorDataset(cfg["train"], cfg["width"], cfg["height"], imgaug = True)
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    # A small worker pool keeps image decoding/augmentation off the training process.
    nw = max(0, opt.workers)
    # 训练集
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   persistent_workers=nw > 0
                                                   )
    #验证集
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 persistent_workers=nw > 0
                                                 )

    # 指定后端设备CUDA&CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 判断是否加载预训练模型
    load_param = False
    premodel_path = cfg["pre_weights"]
    if premodel_path != None and os.path.exists(premodel_path):
        load_param = True

    # 初始化模型结构
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], load_param).to(device)
    summary(model, input_size=(3, cfg["height"], cfg["width"]))

    # 加载预训练模型参数
    if load_param == True:
        load_compatible_pretrained_weights(model, premodel_path, device)
        print("Load finefune model param: %s" % premodel_path)
    else:
        print("Initialize weights: model/backbone/backbone.pth")

    # 构建SGD优化器
    optimizer = optim.SGD(params=model.parameters(),
                          lr=cfg["learning_rate"],
                          momentum=0.949,
                          weight_decay=0.0005,
                          )

    # 学习率衰减策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg["steps"],
                                               gamma=0.1)

    best_ap50_95 = float("-inf")
    best_epoch = None
    os.makedirs("weights", exist_ok=True)

    print('Starting training for %g epochs...' % cfg["epochs"])

    batch_num = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)

        for imgs, targets in pbar:
            # 数据预处理
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            # 模型推理
            preds = model(imgs)
            # loss计算
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device)

            # 反向传播求解梯度
            total_loss.backward()

            #学习率预热
            for g in optimizer.param_groups:
                warmup_num =  5 * len(train_dataloader)
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num/warmup_num, 4)
                    g['lr'] = cfg["learning_rate"] * scale
                    
                lr = g["lr"]

            # 更新模型参数
            if batch_num % cfg["subdivisions"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 打印相关信息
            info = "Epoch:%d LR:%f CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou_loss, obj_loss, cls_loss, total_loss)
            pbar.set_description(info)

            batch_num += 1

        # 模型保存
        if epoch % 10 == 0 and epoch > 0:
            metrics = evaluate_coco(
                val_dataloader, cfg, model, device, cfg["val_annotations"]
            )
            print(
                "Valid COCO AP50-95:%f AP50:%f AP75:%f Detections:%d" % (
                    metrics["ap50_95"], metrics["ap50"], metrics["ap75"],
                    metrics["detections"],
                )
            )

            checkpoint_path = "weights/%s-%d-epoch-%fap50_95-model.pth" % (
                cfg["model_name"], epoch, metrics["ap50_95"]
            )
            torch.save(model.state_dict(), checkpoint_path)
            if metrics["ap50_95"] > best_ap50_95:
                best_ap50_95 = metrics["ap50_95"]
                best_epoch = epoch
                torch.save(model.state_dict(), "weights/best.pt")
                print("Updated weights/best.pt with valid AP50-95:%f" % best_ap50_95)

        # 学习率调整
        scheduler.step()

    print("Training summary: best valid AP50-95:%f at epoch:%d, checkpoint:weights/best.pt" %
          (best_ap50_95, best_epoch))
