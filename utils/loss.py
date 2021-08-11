import math
import torch
import torch.nn as nn
import numpy as np

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
    tcls, tbox, indices, anch = [], [], [], []
    #anchor box数量, 当前batch的标签数量
    anchor_num, label_num = cfg["anchor_num"], targets.shape[0]

    #加载anchor配置
    anchors = np.array(cfg["anchors"])
    anchors = torch.from_numpy(anchors.reshape(len(preds), anchor_num, 2)).to(device)
    
    gain = torch.ones(7, device = device)

    at = torch.arange(anchor_num, device = device).float().view(anchor_num, 1).repeat(1, label_num)
    targets = torch.cat((targets.repeat(anchor_num, 1, 1), at[:, :, None]), 2)

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device = device).float() * g  # offsets

    for i, pred in enumerate(preds):
        #输出特征图的维度
        n, c, h, w = pred.shape

        assert cfg["width"]/w == cfg["height"]/h, "特征图宽高下采样不一致"
        
        #计算下采样倍数
        stride = cfg["width"]/w

        #该尺度特征图对应的anchor配置
        anchors_cfg = anchors[i]/stride

        #将label坐标映射到特征图上
        gain[2:6] = torch.tensor(pred.shape)[[3, 2, 3, 2]]

        gt = targets * gain 

        if label_num:
            #anchor iou匹配
            r = gt[:, :, 4:6] / anchors_cfg[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < 4

            t = gt[j]
            #扩充维度并复制数据
            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors_cfg[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
    
def compute_loss(preds, targets, cfg, device):
    balance = [1.0, 0.4]

    ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])

    #定义obj和cls的损失函数
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))

    cp, cn = smooth_BCE(eps=0.0)
    
    #构建gt
    tcls, tbox, indices, anchors = build_target(preds, targets, cfg, device)

    for i, pred in enumerate(preds):
        pred = pred.reshape(pred.shape[0], cfg["anchor_num"], -1, pred.shape[2], pred.shape[3])
        pred = pred.permute(0, 1, 3, 4, 2)

        tobj = torch.zeros_like(pred[..., 0])  # target obj

        #判断当前batch数据是否有gt
        if len(indices):
            b, a, gj, gi = indices[i]
            nb = b.shape[0]

            if nb:
                ps = pred[b, a, gj, gi]
                
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                ciou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, CIoU=True)  # ciou(prediction, target)
                lbox +=  (1.0 - ciou).mean()

                tobj[b, a, gj, gi] = ciou.detach().clamp(0).type(tobj.dtype)
                
                if ps.size(1) - 5 > 1:
                    t = torch.full_like(ps[:, 5:], cn)  # targets
                    t[range(nb), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pred[..., 4], tobj)  *  balance[i] # obj loss

    lbox *= 3.2
    lobj *= 64
    lcls *= 32
    loss = lbox + lobj + lcls

    return lbox, lobj, lcls, loss
