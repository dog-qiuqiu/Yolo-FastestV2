import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.fpn import *
from model.backbone.shufflenetv2 import *
import nnAudio.features
import cv2

class CQTSpectrogram(nn.Module):
    def __init__(self, cqt_config, width, height, log_scale=True, interpolate=True, convert2image=False):
        super(CQTSpectrogram, self).__init__()
        self.cqt = nnAudio.features.cqt.CQT(sr=cqt_config['sr'], hop_length=cqt_config['hop'], 
                                            fmin=cqt_config['fmin'], n_bins=cqt_config['n_bins'],
                                            bins_per_octave=cqt_config['bins_per_octave'], 
                                            center=False)
        self.width = width
        self.height = height
        self.log_scale = log_scale
        self.interpolate = interpolate
        self.convert2image = convert2image
        
    def forward(self, audio):
        # [1, 1, 176, 150]
        cqt = self.cqt(audio)[:, None]
        if self.interpolate:
            # [1, 1, 192, 160]
            cqt = F.interpolate(cqt, size=(self.height, self.width), mode='bilinear', align_corners=False)
        
        if self.log_scale:
          cqt = torch.log(torch.clamp(cqt, min=1e-7))

        if self.convert2image:
            device = cqt.device
            cqt = cqt.cpu().numpy()[:, 0]
            cqt = cqt - cqt.min(axis=(-1, -2), keepdims=True)
            cqt /= (cqt.max(axis=(-1, -2), keepdims=True) + 1e-6)
            cqt = (cqt * 255).astype('uint8')
            b, h, w = cqt.shape
            cqt = cqt.reshape(-1, w)
            cqt = cv2.applyColorMap(cqt, cv2.COLORMAP_VIRIDIS).astype('float32') / 255.
            cqt = cqt.reshape(-1, h, w, 3).transpose(0, 3, 1, 2)
            cqt = torch.from_numpy(cqt).to(device)

        return cqt

class Detector(nn.Module):
    def __init__(self, cqt_config, width, height, anchor_num, load_param, convert2image=False):
        super(Detector, self).__init__()

        self.cqt = CQTSpectrogram(cqt_config, width, height, convert2image=convert2image)
        out_depth = 72 # 决定fpn模块的通道
        stage_out_channels = [-1, 24, 48, 96, 192]

        in_channel = 3 if convert2image else 1
        self.backbone = ShuffleNetV2(in_channel, stage_out_channels, load_param)
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)

    def forward(self, audio): # [1, 80448]
        # [1, 1, 192, 160]
        cqt = self.cqt(audio)
        C2, C3 = self.backbone(cqt) # [1, 96, 12, 10] [1, 192, 6, 5] # 已经是shufflenetv2第三阶段和第四阶段输出
        # [1, 72, 72, 10] x2 [1, 72, 6, 5] x2
        obj_2, reg_2, obj_3, reg_3 = self.fpn(C2, C3)
        
        # [1, 12, 12, 10]
        out_reg_2 = self.output_reg_layers(reg_2)
        # [1, 3, 12, 10]
        out_obj_2 = self.output_obj_layers(obj_2)

        # [1, 12, 6, 5]
        out_reg_3 = self.output_reg_layers(reg_3)
        # [1, 3, 6, 5]
        out_obj_3 = self.output_obj_layers(obj_3)

        return out_reg_2, out_obj_2, out_reg_3, out_obj_3, cqt

if __name__ == "__main__":
    # test_data = torch.rand(1, 1, 192, 160)
    cqt_config = {
        "sr": 16000,
        "fmin": 27.5,
        "hop": 320,
        "bins_per_octave": 24,
        "n_bins": 176,
        "overlap_ratio": 0.5,
        "duration": 3.0
    }
    width = 160
    height = 192
    model = Detector(cqt_config,width, height, 3, False)
    test_data = torch.rand(1, 80448)
    model(test_data)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "test.onnx",               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization
    


