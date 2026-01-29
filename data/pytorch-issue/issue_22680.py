# torch.rand(B, C, H, W, dtype=...)  # B: batch size, C: 3 (RGB), H: height, W: width
import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.welcome_layer = BasicConv(3, 256, 3, 1, 1)
        self.block_1 = BasicConv(256, 256, 3, 1, 1)
        self.fuck_layers = nn.Sequential(*[BasicConv(256, 256, 3, 1, 1) for _ in range(5)])

    def forward(self, x):
        x = self.welcome_layer(x)
        return self.fuck_layers(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 224, 224  # Example dimensions
    return torch.rand(B, C, H, W, dtype=torch.float32)

