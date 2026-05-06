import os
import cv2
import random
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import Dataset

def contrast_and_brightness(img):
    alpha = random.uniform(0.25, 1.75)
    beta = random.uniform(0.25, 1.75)
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

def motion_blur(image):
    if random.randint(1,2) == 1:
        degree = random.randint(2,3)
        angle = random.uniform(-360, 360)
        image = np.array(image)
    
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred
    else:
        return image

def augment_hsv(img, hgain = 0.0138, sgain = 0.678, vgain = 0.36):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


def random_resize(img):
    h, w, _ = img.shape
    rw = int(w * random.uniform(0.8, 1))
    rh = int(h * random.uniform(0.8, 1))

    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_LINEAR) 
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR) 
    return img

def img_aug(img):
    img = contrast_and_brightness(img)
    #img = motion_blur(img)
    #img = random_resize(img)
    #img = augment_hsv(img)
    return img

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)


def letterbox(img, new_w, new_h, pad_color=(114, 114, 114)):
    """等比缩放到 (new_w, new_h) + 灰边填充 (canvas-style letterbox)。

    返回:
        canvas (np.ndarray, new_h x new_w x 3)
        ratio  (float)        统一缩放比 (长/宽相同)
        pad_l  (int)          左侧填充像素数
        pad_t  (int)          上侧填充像素数
    """
    h, w = img.shape[:2]
    ratio = min(new_w / w, new_h / h)
    rw, rh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_l = (new_w - rw) // 2
    pad_t = (new_h - rh) // 2
    canvas = np.full((new_h, new_w, 3), pad_color, dtype=img.dtype)
    canvas[pad_t:pad_t + rh, pad_l:pad_l + rw] = resized
    return canvas, ratio, pad_l, pad_t


class TensorDataset(Dataset):
    def __init__(self, path, img_size_width=352, img_size_height=352,
                 imgaug=False, keep_ratio=False):
        """
        Args:
            path:            train.txt / valid.txt 路径
            img_size_width:  网络输入宽
            img_size_height: 网络输入高
            imgaug:          是否做训练增强
            keep_ratio:      False=直接 stretch 拉伸 (默认, 适合本数据集 16:9 固定比例);
                             True=letterbox 等比缩放 (适合通用场景, 但本数据集精度较低)
        """
        super().__init__()
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.path = path
        self.data_list = []
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
        self.imgaug = imgaug
        self.keep_ratio = keep_ratio

        # 数据检查
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        # 兼容 YOLO 标准目录结构: images/ 与 labels/ 并列，且文件名中可能包含 "." (如 roboflow 导出)
        base, _ = os.path.splitext(img_path)
        label_path = base.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep) \
                         .replace("/images/", "/labels/") + ".txt"

        # 读图
        img = cv2.imread(img_path)
        ori_h, ori_w = img.shape[:2]

        # 预处理: letterbox 等比 (默认) 或 stretch 直接拉伸
        if self.keep_ratio:
            img, ratio, pad_l, pad_t = letterbox(
                img, self.img_size_width, self.img_size_height
            )
        else:
            img = cv2.resize(
                img, (self.img_size_width, self.img_size_height),
                interpolation=cv2.INTER_LINEAR,
            )
            ratio, pad_l, pad_t = None, 0, 0

        # 数据增强
        if self.imgaug:
            img = img_aug(img)
        img = img.transpose(2, 0, 1)

        # 加载 label 文件 (YOLO 归一化 cx,cy,w,h, 相对原图)
        if not os.path.exists(label_path):
            raise Exception("%s is not exist" % label_path)

        label = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 5:
                    continue
                cls = float(parts[0])
                cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]))

                if self.keep_ratio:
                    # YOLO 归一化坐标 -> 原图像素 -> letterbox 像素 -> 网络输入归一化
                    cx_px = cx * ori_w * ratio + pad_l
                    cy_px = cy * ori_h * ratio + pad_t
                    bw_px = bw * ori_w * ratio
                    bh_px = bh * ori_h * ratio
                    cx = cx_px / self.img_size_width
                    cy = cy_px / self.img_size_height
                    bw = bw_px / self.img_size_width
                    bh = bh_px / self.img_size_height
                # else stretch: 归一化坐标在 [0,1] 范围内不变 (与原仓库行为一致)

                label.append([0, cls, cx, cy, bw, bh])
        label = np.array(label, dtype=np.float32)

        if label.shape[0]:
            assert label.shape[1] == 6, '> 5 label columns: %s' % label_path

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    data = TensorDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)
