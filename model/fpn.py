import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvblock(nn.Module):
    def __init__(self, output_channels, size):
        super(DWConvblock, self).__init__()

        self.block =  nn.Sequential(nn.Conv2d(output_channels, output_channels, size, 1, size // 2, groups = output_channels, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    
                                    nn.Conv2d(output_channels, output_channels, size, 1, size // 2, groups = output_channels, bias = False),
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
        
        # 1x1 5x5 1x1 内外均未改变通道
        self.cls_head_2 = DWConvblock(out_depth, 5)
        self.reg_head_2 = DWConvblock(out_depth, 5)
        
        self.reg_head_3 = DWConvblock(out_depth, 5)
        self.cls_head_3 = DWConvblock(out_depth, 5)

    def forward(self, C2, C3): # [1, 96, 12, 10] [1, 192, 6, 5]
        S3 = self.conv1x1_3(C3) # [1, 72, 6, 5]
        # 前景背景分类共用一个头输出
        cls_3 = self.cls_head_3(S3) # [1, 72, 6, 5]
        obj_3 = cls_3
        # 边界信息单独一个头
        reg_3 = self.reg_head_3(S3) # [1, 72, 6, 5]

        P2 = F.interpolate(C3, scale_factor=2, mode='bilinear') # [1, 192, 12, 10]
        P2 = torch.cat((P2, C2), 1) # [1, 288, 12, 10] 浅层输出的特征还是要结合深层信息
        S2 = self.conv1x1_2(P2) # [1, 72, 12, 10]
        cls_2 = self.cls_head_2(S2) # [1, 72, 12, 10]
        obj_2 = cls_2
        reg_2 = self.reg_head_2(S2) # [1, 72, 12, 10]

        return  obj_2, reg_2, obj_3, reg_3

                     
