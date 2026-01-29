# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
from torch import nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    # normalization along channels
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()  # Ensure the tensor is contiguous
        return x

class TestConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.norm = LayerNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x

class MyModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.test_conv = TestConv(in_channel, out_channel)
        self.pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.test_conv(x)
        x = self.pooling(x)
        return x

def my_model_function():
    return MyModel(2, 1)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((2, 2, 3, 3), dtype=torch.float32).cuda()

