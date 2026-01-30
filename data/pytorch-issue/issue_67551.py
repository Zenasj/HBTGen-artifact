import torch.nn as nn
import math

if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

# Fails despite being able to work correctly
conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), \
    stride=(2, 2), dilation=(1, 1), groups=1, bias=True, padding="same")

import torch
import torch.nn.functional as F


kernel_size=(7, 7)
stride=(2, 2)
dilation=(1, 1)

# Conv2d layer with a stride of 2
conv_layer_s2 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=1, bias=True)

# PyTorch's same padding calculations taken from ConvNd code
_reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
    total_padding = d * (k - 1)
    left_pad = total_padding // 2
    _reversed_padding_repeated_twice[2 * i] = left_pad
    _reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
                    
# Create test input
input_test = torch.zeros(1, 3, 224, 224)

# Pad input like in ConvNd code
input_p = F.pad(input_test, _reversed_padding_repeated_twice)

out_conv = conv_layer_s2(input_p)
print(out_conv.shape)

# Output tensor is expected shape
#>> torch.Size([1, 64, 112, 112])

import torch
import torch.nn.functional as F

class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

conv_layer_s2_same = Conv2dSame(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
out = conv_layer_s2_same(torch.zeros(1, 3, 224, 224))