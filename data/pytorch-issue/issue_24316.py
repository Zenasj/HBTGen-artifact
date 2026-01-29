# torch.rand(1, 3, 512, 512, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn

class _CBNReLU(nn.Module):
    """Convolution plus Batchnorm and Relu"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False):
        super(_CBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class MyModel(nn.Module):

    def __init__(self, num_classes=20):
        super(MyModel, self).__init__()
        self.ds_conv0 = _CBNReLU(3, 32, 3, 2)

    def forward(self, x):
        x = self.ds_conv0(x)
        x = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 512, 512, dtype=torch.float32)

