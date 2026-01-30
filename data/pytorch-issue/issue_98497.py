import torch.nn as nn

import torch

class Filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.resample_filter = torch.rand(4,4)

    def forward(self, x):
        x = torch.nn.functional.pad(x, [1, 1, 1, 1])  # If this line is commented out, it works.
        weight = self.resample_filter[None, None].repeat([x.shape[1]  , 1] + [1] * self.resample_filter.ndim)
        x = torch.nn.functional.conv2d(input=x, padding=1, weight=weight, groups=x.shape[1] )
        return x


x = torch.rand((1, 3, 256, 256))
f = Filter()
y = f(x)
torch.onnx.export(f, x, "test-filter.onnx", opset_version=15)

import torch

class Filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.resample_filter = torch.rand(4,4)

    def forward(self, x):
        old_channel = x.shape[1] # assign channel before padding
        x = torch.nn.functional.pad(x, [1, 1, 1, 1]) 
        weight = self.resample_filter[None, None].repeat([old_channel  , 1] + [1] * self.resample_filter.ndim)
        x = torch.nn.functional.conv2d(input=x, padding=1, weight=weight, groups=old_channel )
        return x


x = torch.rand((1, 3, 256, 256))
f = Filter()
y = f(x)
torch.onnx.export(f, x, "test-filter.onnx", opset_version=16)