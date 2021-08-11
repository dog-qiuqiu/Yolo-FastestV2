import torch
import torch.nn as nn

from model.fpn import *
from model.backbone.shufflenetv2 import *

class Detector(nn.Module):
    def __init__(self, classes, anchor_num, load_param):
        super(Detector, self).__init__()
        out_depth = 112
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.backbone = ShuffleNetV2(stage_out_channels, load_param)
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_layers = nn.Conv2d(out_depth, (5 + classes) * 3, 1, 1, 0, bias=True)

    def forward(self, x):
        C2, C3 = self.backbone(x)
        P2, P3 = self.fpn(C2, C3)
        
        out_2 = self.output_layers(P2)
        out_3 = self.output_layers(P3)
        return out_2, out_3

if __name__ == "__main__":
    model = Detector(80, 3)
    test_data = torch.rand(1, 3, 320, 320)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "test.onnx",               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization
    


