# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyModel(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        # Override padding to 0 since manual padding is applied in forward()
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode,
            device=device, dtype=dtype
        )

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        # Calculate same padding as in TensorFlow
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate padding for height and width dimensions
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self.calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            # Apply padding symmetrically
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2,  # left/right padding
                    pad_h // 2, pad_h - pad_h // 2]  # top/bottom padding
            )
        # Perform convolution with 0 padding (handled by manual padding)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,  # Must be 0 as set in __init__
            self.dilation,
            self.groups,
        )

def my_model_function():
    # Initialize with parameters from the issue's test case
    return MyModel(
        in_channels=3, out_channels=64, kernel_size=(7, 7),
        stride=(2, 2), groups=1, bias=True
    )

def GetInput():
    # Generate input matching the expected dimensions (1x3x224x224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

