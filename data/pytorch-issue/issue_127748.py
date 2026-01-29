# torch.rand(128, 3, 192, 192, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    if isinstance(x, torch.Tensor):
        return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
    else:
        return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

def pad_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int] = (1, 1),
    value: float = 0,
):
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    return x

class MyModel(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding='SAME',
        dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0
    ):
        padding, is_dynamic = 0, True  # Set padding to 0 as per original code
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps  # Overridden by eps parameter passed

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.view(1, self.out_channels, -1), None, None,
            weight=(self.gain * self.scale).view(-1),
            training=True, momentum=0., eps=self.eps
        ).view_as(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def my_model_function():
    return MyModel(3, 16, 3, stride=2, bias=True, eps=1e-5)

def GetInput():
    return torch.rand(128, 3, 192, 192, dtype=torch.float32)

