import os
import cv2
import random
import numpy as np
import math
import torch

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

class TensorDataset():
    def __init__(self, cqt_config, labeldir,  data_dir, min_duration, aug=False):
        sr = cqt_config['sr']
        bins_per_octave = cqt_config['bins_per_octave']
        fmin = cqt_config['fmin']
        hop = cqt_config['hop']
        stride = hop  / sr
        frame = int(cqt_config['duration'] / stride)

        n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1) # 19,855.76
        n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 32768 2.048s
        window_length = (frame - 1) * hop + n_fft # 150 frame 5.028s
        overlap_ratio = cqt_config['overlap_ratio']
        big_hop = int(cqt_config['duration'] * sr * (1 - overlap_ratio))

        self.filepath_num = [] # [filepath, num, data]
        for file in os.listdir(data_dir):
            filepath = os.path.join(data_dir, file)
            audio = np.load(filepath, allow_pickle=True).astype('float32') / 32768
            audio = np.pad(audio, (n_fft//2, 0))
            num = (len(audio) - window_length) // big_hop + 1
            labelpath = os.path.join(labeldir, file.split('.')[0] + '.txt')
            notes = np.loadtxt(labelpath)
            self.filepath_num.append([filepath, num, audio, notes])

        for i in range(1, len(self.filepath_num)):
            self.filepath_num[i][1] += self.filepath_num[i-1][1]
    
        self.frame = frame
        self.n_bins =cqt_config['n_bins']
        self.bins_per_octave = bins_per_octave
        self.aug = aug
        self.big_hop = big_hop
        self.sr = sr
        self.stride = stride
        self.window_length = window_length
        self.n_fft = n_fft
        self.duration = cqt_config['duration']
        self.min_duration = min_duration

    def __getitem__(self, index):

        sample = -1
        pointer = None
        for i in range(len(self.filepath_num)):
            if index < self.filepath_num[i][1]:
                sample = index if i == 0 else index - self.filepath_num[i-1][1]
                pointer = i
                break
        assert sample >= 0, (index, self.filepath_num[-1][1])

        _, _, audio, notes = self.filepath_num[pointer]
        start = sample * self.big_hop
        stop = start + self.window_length
        feature = np.zeros((self.window_length, ), dtype='float32')
        feature[: self.window_length] = audio[start: stop]
        start_t = start / self.sr # 对应帧代表的开头为0s
        stop_t = start_t + self.duration

        boxs = []
        for onset, offset, pitch in notes:
            if onset < stop_t and offset > start_t and pitch >= 21 and pitch <= 108:
                t = min(offset, stop_t) - max(onset, start_t)
                if t >= self.min_duration:
                    x0 = max(onset, start_t) - start_t
                    x0 = x0 / self.stride / self.frame
                    y0 = (pitch - 21) * self.bins_per_octave / 12 / self.n_bins
                    h = 1 - y0
                    w = t / self.stride / self.frame
                    x = x0 + w / 2
                    y = y0 + h / 2
                    boxs.append([0, x, y, w, h]) # 第一维度在collect_fn中被赋值为框序号
        boxs = np.array(boxs, dtype='float32')

        return torch.from_numpy(feature), torch.from_numpy(boxs)

    def __len__(self):
        return self.filepath_num[-1][1]


class TestDataset():
    def __init__(self, cqt_config,  audio_path, aug=False):
        sr = cqt_config['sr']
        bins_per_octave = cqt_config['bins_per_octave']
        fmin = cqt_config['fmin']
        hop = cqt_config['hop']
        stride = hop / sr
        frame = int(cqt_config['duration'] / stride)

        n_fft = sr / fmin / (2** (1 / bins_per_octave) - 1) # 19,855.76
        n_fft = 2 ** int(math.ceil(math.log2(n_fft))) # 32768 2.048s
        window_length = (frame - 1) * hop + n_fft # 150 frame 5.028s
        overlap_ratio = cqt_config['overlap_ratio']
        big_hop = int(cqt_config['duration'] * sr * (1 - overlap_ratio))

        audio = np.load(audio_path, allow_pickle=True).astype('float32') / 32768
        self.audio = np.pad(audio, (n_fft//2, 0))
        self.num = (len(audio) - window_length) // big_hop + 1
    
        self.aug = aug
        self.big_hop = big_hop
        self.window_length = window_length

    def __getitem__(self, index):

        start = index * self.big_hop
        stop = start + self.window_length
        feature = np.zeros((self.window_length, ), dtype='float32')
        feature[: self.window_length] = self.audio[start: stop]

        return torch.from_numpy(feature)
    
    def get_total_audio(self):
        return torch.from_numpy(self.audio)

    def __len__(self):
        return self.num


if __name__ == "__main__":
    import json
    config_path = 'yolo.json'
    with open(config_path) as f:
        config = json.load(f)

    label_dir = os.path.join(config['dataset_dir'], 'label')
    data_dir = os.path.join(config['datadump_dir'], 'train')
    data = TensorDataset(config['cqt'], label_dir, data_dir)
    feature, label = data.__getitem__(1000)
    print(feature.shape)
    print(label.shape)
    print(label)
