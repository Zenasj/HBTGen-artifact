# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class OriginalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_padded = self.reflection_pad(x)
        return self.conv2d(x_padded)

class ModifiedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=reflection_padding)

    def forward(self, x):
        return self.conv2d(x)

class MyModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1):
        super().__init__()
        self.original = OriginalConvLayer(in_channels, out_channels, kernel_size, stride)
        self.modified = ModifiedConvLayer(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        return self.original(x), self.modified(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

