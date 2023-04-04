import math
import torch
import torch.nn as nn
import numpy as np

layer_index = [0, 0, 1, 1]

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def build_target(preds, targets, cfg, device): 
    # preds (128 x 12 x 12 x 10, , 128 x 12 x 6 x 5, ,) 4 # targets (1005, 5)
    tbox, indices, anch = [], [], []
    #anchor box数量, 当前batch的标签数量
    anchor_num, label_num = cfg["m_config"]["anchor_num"], targets.shape[0]

    if label_num == 0:
        return tbox, indices, anch

    #加载anchor配置
    anchors = np.array(cfg["m_config"]["anchors"])
    # [2, 3, 2]
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 2, anchor_num, 2)).to(device)
    
    gain = torch.ones(6, device = device)
    
    # (3, 1005)  # value [0, 1, 2]  repeat(*size) 在每个维度分别repeat，并且在前者基础上进行
    # at可以理解为anchor_targets
    at = torch.arange(anchor_num, device = device).float().view(anchor_num, 1).repeat(1, label_num)
    # (3, 1005, 6) # 前5列为框序号和x0, y0, w, h 最后一列为at
    targets = torch.cat((targets.repeat(anchor_num, 1, 1), at[:, :, None]), 2)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        ], device = device).float() * g  # offsets

    for i, pred in enumerate(preds):
        if i % 2 == 0:  #代表边框信息
            #输出特征图的维度
            _, _, h, w = pred.shape # (12, 10) or (6, 5)

            assert cfg["m_config"]["width"]/w == cfg["m_config"]["height"]/h, "特征图宽高下采样不一致"

            #计算下采样倍数
            stride = cfg["m_config"]["width"]/w # 16 or 32

            #该尺度特征图对应的anchor配置
            anchors_cfg = anchors[layer_index[i]]/stride

            #将label坐标映射到特征图上 这里是宽和高 因为targets是 x0, y0, w, h 先宽后高 需要和targets保持一致
            gain[1:5] = torch.tensor(pred.shape)[[3, 2, 3, 2]] # value [1, 10, 12, 10, 12, 1] or [1, 5, 6, 5, 6, 1] 先特征图的宽，再高

            gt = targets * gain # [3, 1005, 6]

            if label_num:
                #anchor iou匹配
                r = gt[:, :, 3:5] / anchors_cfg[:, None] # 宽 高 比例 # [3, 1005, 2]
                j = torch.max(r, 1. / r).max(2)[0] < 2 # [3, 1005]

                t = gt[j] # [1089, 6]

                #扩充维度并复制数据
                # Offsets
                gxy = t[:, 1:3]  # grid xy [1089, 2]
                gxi = gain[[1, 2]] - gxy  # inverse [1089, 2]
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T # shape 1089, 1089
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T # shape 1089, 1089
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, 1089)
                t = t.repeat((5, 1, 1))[j] # [5, 1089, 6]-->[5308, 6]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # [5308, 2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b = t[:, 0].long().T  # image
            gxy = t[:, 1:3]  # grid xy
            gwh = t[:, 3:5]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 5].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[2] - 1), gi.clamp_(0, gain[1] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors_cfg[a])  # anchors

    return tbox, indices, anch

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

'''
yolov5算法的解析 https://zhuanlan.zhihu.com/p/609264977
'''
def compute_loss(preds, targets, cfg, device):
    # preds [128, 12, 12, 10] x 2 [128, 12, 6, 5] x 2 targets [1005, 5]
    balance = [1.0, 1.0]

    ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
    lbox, lobj = ft([0]), ft([0])

    #定义obj的损失函数
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
    
    #构建gt 其中传入的pred只是使用了shape,并没有使用具体的数值
    tbox, indices, anchors = build_target(preds, targets, cfg, device) # length=2,  anchors shape [5308, 2]

    for i, pred in enumerate(preds):
        #计算reg分支loss
        if i % 2 == 0:
            pred = pred.reshape(pred.shape[0], cfg["m_config"]["anchor_num"], -1, pred.shape[2], pred.shape[3])
            pred = pred.permute(0, 1, 3, 4, 2) # [128, 3, 12, 10, 4] or [128, 3, 6, 5, 4]
            
            #判断当前batch数据是否有gt
            if len(indices):
                b, a, gj, gi = indices[layer_index[i]] # all shapes are [5308, ]
                nb = b.shape[0]

                if nb:
                    ps = pred[b, a, gj, gi] # [5308, 4]                    
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5 # 为框的中心
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[layer_index[i]]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box shape [5308, 4]
                    # shape [5308, ]
                    ciou = bbox_iou(pbox.t(), tbox[layer_index[i]], x1y1x2y2=False, CIoU=True)  # ciou(prediction, target)
                    lbox +=  (1.0 - ciou).mean()

        #计算obj分支loss
        elif i % 2 == 1:
            pred = pred.reshape(pred.shape[0], cfg["m_config"]["anchor_num"], -1, pred.shape[2], pred.shape[3])
            pred = pred.permute(0, 1, 3, 4, 2) # [128, 3, 12, 10, 1] or [128, 3, 6, 5, 1]

            tobj = torch.zeros_like(pred[..., 0])  # target obj [128, 3, 12, 10]

            #判断当前batch数据是否有gt
            if len(indices):
                b, a, gj, gi = indices[layer_index[i]]
                nb = b.shape[0]

                if nb:
                    tobj[b, a, gj, gi] = 1.0
                    
            lobj += BCEobj(pred[..., 0], tobj) * balance[layer_index[i]] # obj loss
        else:
            print("error")
            raise

    loss = lbox + lobj

    return lbox, lobj, loss
