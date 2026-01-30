import torch.nn as nn

import torch

class FunctionalConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

    def forward(self, x, weight, bias):
        return torch.nn.functional.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FunctionalConv2d()
        self.conv2 = FunctionalConv2d()

    def forward(self, x, weight, bias):
        x = self.conv1(x, weight, bias)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, weight, bias)
        return x

inputs = (torch.randn(1, 3, 5, 5), torch.randn(3, 3, 3, 3), torch.rand(3))
gm, _ = torch._dynamo.export(M(), aten_graph=True)(*inputs)