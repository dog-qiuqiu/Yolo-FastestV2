import time

import torch
import torchvision
import torch.nn.functional as F

import os, time
import numpy as np
from tqdm import tqdm
import cv2

#加载data
def load_datafile(data_path):
    #需要配置的超参数
    cfg = {"model_name":None,
    
           "epochs": None,
           "steps": None,           
           "batch_size": None,
           "subdivisions":None,
           "learning_rate": None,

           "pre_weights": None,        
           "classes": None,
           "width": None,
           "height": None,           
           "anchor_num": None,
           "anchors": None,

           "val": None,           
           "train": None,
           "names":None
        }

    assert os.path.exists(data_path), "请指定正确配置.data文件路径"

    #指定配置项的类型
    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "val", "train", "names", "pre_weights"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                   "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]
    
    #加载配置文件
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            else:
                data = line.strip().split("=")
                #配置项类型转换
                if data[0] in cfg:
                    if data[0] in int_type_key:
                       cfg[data[0]] = int(data[1])
                    elif data[0] in str_type_key:
                        cfg[data[0]] = data[1]
                    elif data[0] in float_type_key:
                        cfg[data[0]] = float(data[1])
                    elif data[0] in list_type_key:
                        cfg[data[0]] = [float(x) for x in data[1].split(",")]
                    else:
                        print("配置文件有错误的配置项")
                else:
                    print("%s配置文件里有无效配置项:%s"%(data_path, data))
    return cfg

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, n_gt):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        n_gt: gt num
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # tp shape (38400,)  conf shape (38400,)

    # Sort by objectness
    i = np.argsort(-conf) # i是置信度从大到小的排列序号
    tp, conf = tp[i], conf[i] # 排好序

    # Create Precision-Recall curve and compute AP for each class
    n_p = len(tp)  # Number of predicted objects
    ap, p, r = [], [], []
    if n_p == 0 and n_gt == 0:
        pass
    elif n_p == 0 or n_gt == 0:
        ap.append(0)
        r.append(0)
        p.append(0)
    elif n_p != 0 and n_gt != 0:
        # Accumulate FPC and TPs
        fpc = (1 - tp[i]).cumsum()
        tpc = (tp[i]).cumsum()

        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        r.append(recall_curve[-1])

        # Precision
        precision_curve = tpc / (n_p + 1e-16)
        p.append(precision_curve[-1])

        # AP from recall-precision curve
        ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return np.mean(p), np.mean(r), np.mean(ap), np.mean(f1)

def get_batch_statistics(outputs, targets, iou_threshold, device):
    """ Compute true positives, predicted scores and predicted labels per sample """
    # targets shape [641, 5]
    batch_metrics = []
    for sample_i in range(len(outputs)): # length 128

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # [300, 5] # 每一个样本中框的集合
        pred_boxes = output[:, :4] # [300, 4]
        pred_scores = output[:, 4] # [300, ]

        true_positives = np.zeros(pred_boxes.shape[0]) # [300, ] # 预测框one hot其中正确的位置被标记为1，其他保持0

        annotations = targets[targets[:, 0] == sample_i][:, 1:] # 3个锚框 [3, 4]

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations

            for pred_i, pred_box in enumerate(pred_boxes):
                
                pred_box = pred_box.to(device)

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores])
    return batch_metrics

def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Inputs:
         prediction: (x0, y0, w, h)
    Returns:
         detections with shape: nx5 (x1, y1, x2, y2, conf)
    """
    # prediction shape [128, 450, 5]

    # Settings
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after

    t = time.time()
    output = [torch.zeros((0, 5), device="cpu")] * prediction.shape[0] # shape [0, 5] x 128 0位置让这个维度可以放任意维度的矩阵

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box, conf = xywh2xyxy(x[:, :4]), x[:, 4:]        
        x = torch.cat((box, conf), 1)

        # Detections matrix nx5 (xyxy, conf)  x shape [450, 5]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        # boxes (offset by class), scores
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def make_grid(h, w, cfg, device="cpu"):
    hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((wv, hv), 2).repeat(1,1,3).reshape(h, w, cfg["m_config"]["anchor_num"], -1).to(device)

#特征图后处理
def handel_preds(preds, cfg, device):
    #加载anchor配置
    anchors = np.array(cfg["m_config"]["anchors"])
    # [2, 3, 2]
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 2, cfg["m_config"]["anchor_num"], 2)).to(device)
    output_bboxes = []

    for i in range(len(preds) // 2): # 0~2
        batch_bboxes = []
        reg_preds = preds[i * 2]
        obj_preds = preds[(i * 2) + 1]

        for index, r, o in zip(range(len(reg_preds)), reg_preds, obj_preds): # 在batch维度循环
            # [12, 10, 12]
            r = r.permute(1, 2, 0)
            # [12, 10, 3, 4]
            r = r.reshape(r.shape[0], r.shape[1], cfg["m_config"]["anchor_num"], -1)

            # [12, 10, 3]
            o = o.permute(1, 2, 0)
            # [12, 10, 3, 1]
            o = o.reshape(o.shape[0], o.shape[1], cfg["m_config"]["anchor_num"], -1)

            # [12, 10, 3, 5] # 前两个为高和宽的尺寸,第三个为anchor_num, 最后为xywh+conf
            anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + 1)

            #计算anchor box的cx, cy
            # [12, 10, 3, 2]
            grid = make_grid(r.shape[0], r.shape[1], cfg, device)
            stride = cfg["m_config"]["height"] /  r.shape[0] # value 16.0 or 32.0
            anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride

            #计算anchor box的w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg # wh

            #计算obj分数
            anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()

            #torch tensor 转为 numpy array
            anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            batch_bboxes.append(anchor_boxes)     

        #n, anchor num, h, w, box => n, (anchor num*h*w), box
        batch_bboxes = torch.from_numpy(np.array(batch_bboxes)) # [128, 12, 10, 3, 5]
        batch_bboxes = batch_bboxes.view(batch_bboxes.shape[0], -1, batch_bboxes.shape[-1]) # [128, 360, 5]

        output_bboxes.append(batch_bboxes)    
        
    #merge 两个特征图上的检测结果合并，其中第一个图预测结果更多，因为尺寸更大
    output = torch.cat(output_bboxes, 1) # [128, 450, 5]
            
    return output

#特征图后处理trt
def handel_preds_trt(preds, cfg):
    #加载anchor配置
    anchors = np.array(cfg["m_config"]["anchors"])
    # [2, 3, 2]
    anchors = torch.from_numpy(anchors.reshape(len(preds), cfg["m_config"]["anchor_num"], 2))
    output_bboxes = []

    for i in range(len(preds)): # 0~2
        batch_bboxes = []
        reg_preds = preds[i][..., :12]
        obj_preds = preds[i][..., 12: ]

        for index, r, o in zip(range(len(reg_preds)), reg_preds, obj_preds): # 在batch维度循环
            # [12, 10, 3, 4]
            r = r.reshape(r.shape[0], r.shape[1], cfg["m_config"]["anchor_num"], -1)
            # [12, 10, 3, 1]
            o = o.reshape(o.shape[0], o.shape[1], cfg["m_config"]["anchor_num"], -1)

            # [12, 10, 3, 5] # 前两个为高和宽的尺寸,第三个为anchor_num, 最后为xywh+conf
            anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + 1)

            #计算anchor box的cx, cy
            # [12, 10, 3, 2]
            grid = make_grid(r.shape[0], r.shape[1], cfg)
            stride = cfg["m_config"]["height"] /  r.shape[0] # value 16.0 or 32.0
            anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2]* 2. - 0.5) + grid) * stride

            #计算anchor box的w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4] * 2) ** 2 * anchors_cfg # wh

            #计算obj分数
            anchor_boxes[:, :, :, 4] = o[:, :, :, 0]

            #torch tensor 转为 numpy array
            anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            batch_bboxes.append(anchor_boxes)     

        #n, anchor num, h, w, box => n, (anchor num*h*w), box
        batch_bboxes = torch.from_numpy(np.array(batch_bboxes)) # [128, 12, 10, 3, 5]
        batch_bboxes = batch_bboxes.view(batch_bboxes.shape[0], -1, batch_bboxes.shape[-1]) # [128, 360, 5]

        output_bboxes.append(batch_bboxes)    
        
    #merge 两个特征图上的检测结果合并，其中第一个图预测结果更多，因为尺寸更大
    output = torch.cat(output_bboxes, 1) # [128, 450, 5]
            
    return output

#模型评估
def evaluation(val_dataloader, cfg, model, device, conf_thres = 0.01, nms_thresh = 0.4, iou_thres = 0.5):
    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    pbar = tqdm(val_dataloader)

    gt_num = 0
    for imgs, targets in pbar: # [128, 80448] [641, 5]
        imgs = imgs.to(device)
        targets = targets.to(device)
        gt_num += len(targets)    
        
        # Rescale target
        targets[:, 1:] = xywh2xyxy(targets[:, 1:])
        targets[:, 1:] *= torch.tensor([cfg["m_config"]["width"], cfg["m_config"]["height"], 
                                        cfg["m_config"]["width"], cfg["m_config"]["height"]]).to(device)

        #对预测的anchorbox进行nms处理
        with torch.no_grad():
            preds = model(imgs)[: -1]

            #特征图后处理:生成anchorbox
            output = handel_preds(preds, cfg, device)
            output_boxes = non_max_suppression(output, conf_thres = conf_thres, iou_thres = nms_thresh)

        sample_metrics += get_batch_statistics(output_boxes, targets, iou_thres, device) # 每批数据的预测结果的情况追加
        pbar.set_description("Evaluation model:") 

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] # 将一个batch的预测结果整合
    metrics_output = ap_per_class(true_positives, pred_scores, gt_num)
    
    return metrics_output

def dump_data(features, dump_dir, total_boxs=None, confs=None, rgb=False):

    os.makedirs(dump_dir, exist_ok=True)
    height, width = features.shape[-2: ]
    resize_width, resize_height = width * 2, height * 2
    imgs = []
    for index, img in enumerate(features):
        if rgb:
            img = img.transpose(1, 2, 0)
            if width < 300:
                img = cv2.resize(img, (resize_width, resize_height))
            img = (img * 255).astype('uint8')
        else:
            img = img[0]
            if width < 300:
                img = cv2.resize(img, (resize_width, resize_height))
            img = img - img.min()
            img /= (img.max() + 1e-6)
            img = (img * 255).astype('uint8')
            img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        imgs.append(img)

    if not isinstance(total_boxs, list):
        total_boxs = total_boxs.tolist()
    total_boxs.sort(key=lambda x: (x[0], x[1]))
    
    height, width = imgs[0].shape[: -1]
    for index, (img_index, x0, y0, w, h) in enumerate(total_boxs):
        x1, y1 = max(0, x0 - w / 2), max(0, y0 - h / 2)
        x2, y2 = min(1, x0 + w / 2), max(1, y0 + h / 2)
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height) - 1
        imgs[int(img_index)] = cv2.rectangle(imgs[int(img_index)], (x1, y1), (x2, y2), (0, 0, 255))
        if confs:
            imgs[int(img_index)] = cv2.putText(imgs[int(img_index)], '{:.2f}'.format(confs[index]), (x1, y1 - 5 if index % 2 == 0 else y1 + 12), 0, 0.4, (0, 0, 255))

    for index, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dump_dir, 'cqt_{}.png'.format(index)), imgs[index])

def dump_test_data(feature, dumppath, total_notes, cqt_config, no=''):

    height, width = feature.shape
    resize_width, resize_height = width * 2, height * 2
    feature = cv2.resize(feature, (resize_width, resize_height))
    feature = feature - feature.min()
    feature /= (feature.max() + 1e-6)
    feature = (feature * 255).astype('uint8')
    img = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)
    time_length = width * cqt_config['hop'] / cqt_config['sr']
    n_bins = cqt_config['n_bins']

    for index, (onset, offset, pitch, conf) in enumerate(total_notes):
        x1 = int(onset * resize_width / time_length)
        x2 = int(offset * resize_width / time_length)
        y1 = int((pitch - 21) * cqt_config['bins_per_octave'] / 12 / n_bins * resize_height)
        y2 = resize_height - 1
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
        img = cv2.putText(img, '{:.2f}'.format(conf), (x1, y1 - 5 if index % 2 == 0 else y1 + 12), 0, 0.4, (0, 0, 255))

    cv2.imwrite(dumppath, img)

def process_one_image_boxes(output_box):
    if len(output_box) < 3:
        return output_box
    new_output_box = [output_box[0]]
    for i in range(1, len(output_box) - 1):
        if output_box[i][0] < output_box[i-1][2] and output_box[i][2] > output_box[i+1][0]:
            continue
        else:
            new_output_box.append(output_box[i])
    new_output_box.append(output_box[-1])
    return new_output_box

def convert_boxs_to_notes(output_boxes, hop, scale_h, scale_w, width):
    notes = []
    last_state = 0 # 0 for end 1 for open
    last_pos = 0
    for i, output_box in enumerate(output_boxes):
        offset_pixel = hop * i
        output_box = output_box.cpu().numpy().tolist()
        output_box.sort(key=lambda x: x[0])
        output_box = process_one_image_boxes(output_box)
        
        first = True
        for j, (x1, y1, x2, y2, conf) in enumerate(output_box):
            if x1 < last_pos - 12 and x2 < last_pos + 7:
                continue
            onset = (x1 + offset_pixel) * scale_w
            offset = (x2 + offset_pixel) * scale_w
            pitch = y1 * scale_h + 21.0
            # 上一个音符open #或者 上一个音符open但是last_state有误
            if first and last_state and x1 <= last_pos + 7:# or x1 < last_pos and x2 > last_pos):
                # if len(notes) > 1 and notes[-1][0] < notes[-2][1]  and j > 0 and output_box[j][0] > output_box[j-1][2]:
                #     notes[-1][0] = onset
                notes[-1][1] = offset
                notes[-1][2] = (notes[-1][2] + pitch) / 2
                notes[-1][-1] = (notes[-1][-1] + conf) / 2
                first = False
            else:
                notes.append([onset, offset, pitch, conf])

            if j == len(output_box) - 1:
                last_state = 0 if x2 < width - 4 else 1
                last_pos = x2 - hop if x2 < width - 4 else x1 - hop
            
    return notes

class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for _ in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas, dtype='float32')

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    mixup_lambda = torch.from_numpy(mixup_lambda).to(x.device)
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

def do_targets_mixup(targets, mixup_lambda):
    outs = []
    _targets = targets.clone()
    for target in _targets:
        if mixup_lambda[int(target[0])] > 0.3:
            target[0] = int(target[0]) // 2
            outs.append(target)
    outs = torch.stack(outs, 0)
    return outs