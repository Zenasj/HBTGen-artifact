import torch.nn as nn

import torch
from torch import nn

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 4, 1))

    def forward(self, x):
        return self.net(x)

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.subnet = nn.Sequential(nn.Conv2d(4, 4, 1))
        self.net = nn.Sequential(self.subnet)

    def forward(self, x):
        return self.net(x)

dummy_input = torch.Tensor(1, 4, 256, 256)

print('Started conversion of Net1')
torch.onnx.export(Net1(), dummy_input, 'net1.proto')
print('Finished conversion of Net1')

print('Started conversion of Net2')
torch.onnx.export(Net2(), dummy_input, 'net2.proto')
print('Finished conversion of Net2')

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Sequential(nn.Conv2d(4, 4, 1)))

    def forward(self, x):
        return self.net(x)