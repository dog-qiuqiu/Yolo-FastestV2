import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        # 实现了两个类型的模块，不降采样和降采样的
        # 不降采样的通道不变（前置channel split）降采样的通道翻倍
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

class ShuffleNetV2(nn.Module):
    def __init__(self, in_channel, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        # shufflenetv2开头已经降采样了两次
        # 用作降采样
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channel, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        # 继续降采样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stageSeq = []
            for i in range(numrepeat):
                # 每一个阶段的第一块先降采样，并将通道扩充两倍，其他块则保持尺寸不变
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))
        
        if load_param == True:
            print("load param...")
            self._initialize_weights(in_channel)

    def forward(self, x): # [1, 1, 192, 160]
        x = self.first_conv(x) # [1, 24, 96, 80]
        x = self.maxpool(x) # [1, 24, 48, 40]
        C1 = self.stage2(x) # [1, 48, 24, 20]
        C2 = self.stage3(C1) # [1, 96, 12, 10]
        C3 = self.stage4(C2) # [1, 192, 6, 5]

        return C2, C3
    
    # 部分参数加载方法参考 https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    def _initialize_weights(self, in_channel):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load("./model/backbone/backbone.pth", map_location=device)

        if in_channel == 3:
            self.load_state_dict(pretrained_dict, strict = True)
        else:
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('first_conv')}
            # 2. add the missing keys
            model_dict = {k: v for k, v in self.state_dict().items() if k.startswith('first_conv')}
            # 3. load the new state dict
            self.load_state_dict({**pretrained_dict, **model_dict})


if __name__ == "__main__":
    model = ShuffleNetV2()
    print(model)
    test_data = torch.rand(1, 3, 320, 320)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())
