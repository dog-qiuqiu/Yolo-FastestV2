import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvblock(nn.Module):
    def __init__(self, input_channels, output_channels, size):
        super(DWConvblock, self).__init__()
        self.size = size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block =  nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(output_channels, output_channels, size, 1, 2, groups = output_channels, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels, output_channels // 2, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels // 2),
                                    
                                    nn.Conv2d(output_channels // 2, output_channels // 2, size, 1, 2, groups = output_channels // 2, bias = False),
                                    nn.BatchNorm2d(output_channels // 2),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels // 2, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    )
                                    
    def forward(self, x):
        x = self.block(x)
        return x

class LightFPN(nn.Module):
    def __init__(self, input2_depth, input3_depth, out_depth):
        super(LightFPN, self).__init__()
        
        self.head_2 = DWConvblock(input2_depth, out_depth, 5)
        self.head_3 = DWConvblock(input3_depth, out_depth, 5)

    def forward(self, C2, C3):
        out_3 = self.head_3(C3)

        P2 = F.interpolate(C3, scale_factor=2)
        P2 = torch.cat((P2, C2),1)
        out_2 = self.head_2(P2)

        return  out_2, out_3

                     
