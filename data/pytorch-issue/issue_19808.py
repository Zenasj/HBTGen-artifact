import torch.nn as nn

import torch
from torch import nn

class mylayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 32, 1, 1, bias=False)

    def forward(self, *input):
        x1, x2 = input
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1, y2

td = torch.rand(1, 3, 32, 32).cuda()
model = mylayer().cuda()
print(model(td, td))

seq = nn.Sequential(mylayer(), mylayer())
out = seq(td, td)

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

class n_to_n(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1, y2


class n_to_one(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        return y1 + y2


class one_to_n(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 3, 1, 1, bias=False)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        return y1, y2

seq = mySequential(one_to_n(), n_to_n(), n_to_one()).cuda()
td = torch.rand(1, 3, 32, 32).cuda()

out = seq(td)
print(out.size())

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs