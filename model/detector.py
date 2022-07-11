import torch
import torch.nn as nn

from model.fpn import *
from model.backbone.shufflenetv2 import *

import torch.nn.functional as F

class Detector(nn.Module):
    def __init__(self, classes, anchor_num, load_param, export_onnx = False):
        super(Detector, self).__init__()
        out_depth = 72
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.export_onnx = export_onnx
        self.backbone = ShuffleNetV2(stage_out_channels, load_param)
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

    def forward(self, x):
        C2, C3 = self.backbone(x)
        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(C2, C3)
        
        out_reg_2 = self.output_reg_layers(reg_2)
        out_obj_2 = self.output_obj_layers(obj_2)
        out_cls_2 = self.output_cls_layers(cls_2)

        out_reg_3 = self.output_reg_layers(reg_3)
        out_obj_3 = self.output_obj_layers(obj_3)
        out_cls_3 = self.output_cls_layers(cls_3)
        
        if self.export_onnx:
            print("export onnx ...")
            out0 = out_reg_2.permute(0, 2, 3, 1)   ###去掉batchs维度
            out0 = out0.view(-1, out_reg_2.shape[1])
            out1 = out_obj_2.permute(0, 2, 3, 1)
            out1 = out1.view(-1, out_obj_2.shape[1])
            out2 = out_cls_2.permute(0, 2, 3, 1)
            out2 = out2.view(-1, out_cls_2.shape[1])
            
            out_reg_2 = out0.sigmoid()
            out_obj_2 = out1.sigmoid()
            out_cls_2 = F.softmax(out2, dim = 1)
            print("out_reg_2: ", out_reg_2.shape)
            print("out_obj_2: ", out_obj_2.shape)
            print("out_cls_2: ", out_cls_2.shape)
            
            tensor1 = torch.cat((out_reg_2, out_obj_2, out_cls_2),1)
            print("tensor1: ", tensor1.shape)

            out3 = out_reg_3.permute(0, 2, 3, 1)   ###去掉batchs维度
            out3 = out3.view(-1, out_reg_3.shape[1])
            out4 = out_obj_3.permute(0, 2, 3, 1)
            out4 = out4.view(-1, out_obj_3.shape[1])
            out5 = out_cls_3.permute(0, 2, 3, 1)
            out5 = out5.view(-1, out_cls_3.shape[1])

            out_reg_3 = out3.sigmoid()
            out_obj_3 = out4.sigmoid()
            out_cls_3 = F.softmax(out5, dim = 1)
            print("out_reg_3: ", out_reg_3.shape)
            print("out_obj_3: ", out_obj_3.shape)
            print("out_cls_3: ", out_cls_3.shape)
            tensor2 = torch.cat((out_reg_3, out_obj_3, out_cls_3),1)
            print("tensor2: ", tensor2.shape)

            output = torch.cat((tensor1, tensor2), 0)
            print("output: ", output.shape)
            return output
        else:
            return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3

if __name__ == "__main__":
    model = Detector(80, 3, False)
    test_data = torch.rand(1, 3, 352, 352)
    # onnx [通用]的神经网络交换格式，用不同的框架编写的模型可以在不同的平台中流通。
    torch.onnx.export(model,                    # model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "test.onnx",               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization
