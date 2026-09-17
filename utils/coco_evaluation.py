"""COCO bbox evaluation for Yolo-FastestV2 predictions."""

from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from utils.utils import handel_preds, non_max_suppression

COCO_CONF_THRESH = 0.001
COCO_NMS_IOU_THRESH = 0.6


def evaluate_coco(val_dataloader, cfg, model, device, annotation_path,
                  conf_thres=COCO_CONF_THRESH, nms_thresh=COCO_NMS_IOU_THRESH):
    """Evaluate a sequential loader with COCOeval and fixed YOLO-style postprocessing."""
    if not isinstance(val_dataloader.sampler, SequentialSampler):
        raise ValueError("COCO evaluation requires a DataLoader with shuffle=False")

    coco_gt = COCO(annotation_path)
    images_by_name = {image["file_name"]: image for image in coco_gt.dataset["images"]}
    image_paths = val_dataloader.dataset.data_list
    image_names = [Path(image_path).name for image_path in image_paths]
    if set(image_names) != set(images_by_name):
        raise ValueError("Dataset image list does not match COCO annotation images")

    detections = []
    image_offset = 0
    model.eval()

    with torch.no_grad():
        for imgs, _ in tqdm(val_dataloader, desc="COCO evaluation"):
            batch_size = imgs.shape[0]
            batch_names = image_names[image_offset:image_offset + batch_size]
            image_offset += batch_size

            preds = model(imgs.to(device).float() / 255.0)
            decoded = handel_preds(preds, cfg, device)
            output_boxes = non_max_suppression(
                decoded, conf_thres=conf_thres, iou_thres=nms_thresh
            )

            for image_name, boxes in zip(batch_names, output_boxes):
                image = images_by_name[image_name]
                scale_x = image["width"] / cfg["width"]
                scale_y = image["height"] / cfg["height"]

                for x1, y1, x2, y2, score, category_id in boxes.tolist():
                    x1 = max(0.0, min(x1 * scale_x, image["width"]))
                    y1 = max(0.0, min(y1 * scale_y, image["height"]))
                    x2 = max(0.0, min(x2 * scale_x, image["width"]))
                    y2 = max(0.0, min(y2 * scale_y, image["height"]))
                    width, height = x2 - x1, y2 - y1
                    if width > 0.0 and height > 0.0:
                        detections.append({
                            "image_id": image["id"],
                            "category_id": int(category_id),
                            "bbox": [x1, y1, width, height],
                            "score": score,
                        })

    if not detections:
        print("COCO evaluation: no detections after confidence filtering and NMS")
        return {"ap50_95": 0.0, "ap50": 0.0, "ap75": 0.0, "detections": 0}

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = [images_by_name[name]["id"] for name in image_names]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "ap50_95": float(coco_eval.stats[0]),
        "ap50": float(coco_eval.stats[1]),
        "ap75": float(coco_eval.stats[2]),
        "detections": len(detections),
    }
