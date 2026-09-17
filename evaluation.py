"""Evaluate one Yolo-FastestV2 checkpoint with standard COCO bbox metrics."""

import argparse
import os

import torch

from model.detector import Detector
import utils.datasets
import utils.utils
from utils.coco_evaluation import evaluate_coco


def get_split_paths(cfg, split):
    """Return the image list and COCO annotation file for a named split."""
    if split == "valid":
        return cfg["val"], cfg["val_annotations"]
    return cfg["test"], cfg["test_annotations"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Training profile *.data")
    parser.add_argument("--weights", type=str, required=True,
                        help="Checkpoint .pth or best.pt")
    parser.add_argument("--split", choices=("valid", "test"), default="valid",
                        help="Dataset split to evaluate")
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    dataset_path, annotation_path = get_split_paths(cfg, opt.split)

    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(dataset_path), "请指定正确的数据集清单路径"
    assert os.path.exists(annotation_path), "请指定正确的COCO标注路径"

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    workers = 0
    dataset = utils.datasets.TensorDataset(dataset_path, cfg["width"], cfg["height"], imgaug=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.datasets.collate_fn,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    print("Evaluation split:%s" % opt.split)
    print("COCO annotations:%s" % annotation_path)
    metrics = evaluate_coco(dataloader, cfg, model, device, annotation_path)
    print(
        "COCO AP50-95:%f AP50:%f AP75:%f Detections:%d" % (
            metrics["ap50_95"], metrics["ap50"], metrics["ap75"],
            metrics["detections"],
        )
    )
