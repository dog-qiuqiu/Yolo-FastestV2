import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvblock(nn.Module):
    def __init__(self, input_channels, output_channels, size):
        super(DWConvblock, self).__init__()
        self.size = size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block =  nn.Sequential(nn.Conv2d(output_channels, output_channels, size, 1, 2, groups = output_channels, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    
                                    nn.Conv2d(output_channels, output_channels, size, 1, 2, groups = output_channels, bias = False),
                                    nn.BatchNorm2d(output_channels ),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    )
                                    
    def forward(self, x):
        x = self.block(x)
        return x

class LightFPN(nn.Module):
    def __init__(self, input2_depth, input3_depth, out_depth):
        super(LightFPN, self).__init__()

        self.conv1x1_2 = nn.Sequential(nn.Conv2d(input2_depth, out_depth, 1, 1, 0, bias = False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

        self.conv1x1_3 = nn.Sequential(nn.Conv2d(input3_depth, out_depth, 1, 1, 0, bias = False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )
        
        self.cls_head_2 = DWConvblock(input2_depth, out_depth, 5)
        self.reg_head_2 = DWConvblock(input2_depth, out_depth, 5)
        
        self.reg_head_3 = DWConvblock(input3_depth, out_depth, 5)
        self.cls_head_3 = DWConvblock(input3_depth, out_depth, 5)

    def forward(self, C2, C3):
        S3 = self.conv1x1_3(C3)
        cls_3 = self.cls_head_3(S3)
        obj_3 = cls_3
        reg_3 = self.reg_head_3(S3)

        P2 = F.interpolate(C3, scale_factor=2)
        P2 = torch.cat((P2, C2),1)
        S2 = self.conv1x1_2(P2)
        cls_2 = self.cls_head_2(S2)
        obj_2 = cls_2
        reg_2 = self.reg_head_2(S2)

        return  cls_2, obj_2, reg_2, cls_3, obj_3, reg_3

                     
