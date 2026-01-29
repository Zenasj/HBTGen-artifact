# torch.rand(B, C, D, H, W, dtype=torch.float32)  # e.g., 2x3x256x256x256
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClampedConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', **kwargs):
        self.fp16 = None
        if 'fp16' in kwargs:
            self.fp16 = kwargs['fp16']
            kwargs.pop('fp16')
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[2] + 1) // 2, self.padding[2] // 2,  # Width padding
                (self.padding[1] + 1) // 2, self.padding[1] // 2,  # Height padding
                (self.padding[0] + 1) // 2, self.padding[0] // 2   # Depth padding
            )
            output = F.conv3d(
                F.pad(input, list(expanded_padding), mode='circular'),
                self.weight, self.bias, self.stride, self.padding,  # Corrected padding parameter
                self.dilation, self.groups
            )
        else:
            output = F.conv3d(
                input, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups
            )
        if self.fp16:
            output = torch.clamp(output, -6.55e4, 6.55e4)
        return output

class MyModel(nn.Module):
    def __init__(self, nc=4, fp16=True):
        super().__init__()
        self.fp16 = fp16
        self.conv1 = ClampedConv3d(3, 16, kernel_size=3, padding=1, fp16=fp16)
        self.relu1 = nn.ReLU()
        self.conv2 = ClampedConv3d(16, 32, kernel_size=3, padding=1, fp16=fp16)
        self.relu2 = nn.ReLU()
        self.conv_out = ClampedConv3d(32, nc, kernel_size=1, fp16=fp16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv_out(x)
        return x

def my_model_function():
    return MyModel(nc=4, fp16=True)

def GetInput():
    return torch.rand(2, 3, 256, 256, 256, dtype=torch.float32)

