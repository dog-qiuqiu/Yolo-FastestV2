import copy

import torch
import torch.nn as nn
from torchsummary import summary

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
        if self.stride==1:
            x_proj, x = self.channel_shuffle_4d(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle_4d(self, x):
        """Split adjacent channel pairs into the projection and main branches.

        The original implementation reshaped to 3D and 5D, transposed, and
        indexed the result.  Selecting even and odd channels directly has the
        same channel order while exporting as 4D ONNX Slice nodes.
        """
        input_channels = self.inp * 2
        assert input_channels % 4 == 0
        projection_channels = [
            x[:, channel:channel + 1, :, :] for channel in range(0, input_channels, 2)
        ]
        main_channels = [
            x[:, channel:channel + 1, :, :] for channel in range(1, input_channels, 2)
        ]
        return torch.cat(projection_channels, dim=1), torch.cat(main_channels, dim=1)


class ShuffleV2ExportBlock(nn.Module):
    """Export-only stride-one block that replaces channel shuffle with 1x1 convolutions."""

    def __init__(self, source_block):
        super().__init__()
        if source_block.stride != 1:
            raise ValueError("ShuffleV2ExportBlock only supports stride-one blocks")

        self.branch_main = copy.deepcopy(source_block.branch_main)
        original_conv = self.branch_main[0]
        input_channels = original_conv.in_channels * 2
        self.projection_select = nn.Conv2d(input_channels, original_conv.in_channels, 1, bias=False).to(
            device=original_conv.weight.device, dtype=original_conv.weight.dtype
        )
        with torch.no_grad():
            self.projection_select.weight.zero_()
            channels = torch.arange(original_conv.in_channels, device=original_conv.weight.device)
            self.projection_select.weight[channels, channels * 2, 0, 0] = 1.0
        self.projection_select.weight.requires_grad_(False)

        expanded_conv = nn.Conv2d(input_channels, original_conv.out_channels, 1, bias=original_conv.bias is not None).to(
            device=original_conv.weight.device, dtype=original_conv.weight.dtype
        )
        with torch.no_grad():
            expanded_conv.weight.zero_()
            expanded_conv.weight[:, 1::2, :, :] = original_conv.weight
            if original_conv.bias is not None:
                expanded_conv.bias.copy_(original_conv.bias)
        self.branch_main[0] = expanded_conv

    def forward(self, x):
        """Run both original ShuffleNetV2 branches without layout operations."""
        return torch.cat((self.projection_select(x), self.branch_main(x)), 1)


def replace_shuffle_blocks_for_export(module):
    """Recursively replace stride-one ShuffleNetV2 blocks in an export model."""
    for child_name, child_module in module.named_children():
        if isinstance(child_module, ShuffleV2Block) and child_module.stride == 1:
            setattr(module, child_name, ShuffleV2ExportBlock(child_module))
        else:
            replace_shuffle_blocks_for_export(child_module)
    return module

class ShuffleNetV2(nn.Module):
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))
        
        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)

        return C2, C3

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load("./model/backbone/backbone.pth", map_location=device), strict = True)

if __name__ == "__main__":
    model = ShuffleNetV2()
    print(model)
    test_data = torch.rand(1, 3, 320, 320)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())
