# torch.rand(B, C, T, H, W, dtype=torch.float32)  # B: batch size, C: channels, T: time (frames), H: height, W: width

import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        conv_a_stride = (2, 1, 1)
        conv_b_stride = (1, 2, 2)
        stride = tuple(map(int, map(np.prod, zip(conv_a_stride, conv_b_stride))))
        self.conv = nn.Conv3d(3, 2, (3, 5, 2), stride=stride, padding=(3, 2, 0), bias=False)
        self.resblock = ResBlock()
        self.net = Net(self.resblock)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock(x)
        x = self.net(x)
        return x

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.branch2 = IdentityModel()
        self.branch_fusion = self.branch_fusion_func

    def branch_fusion_func(self, x, y):
        return x + y

    def forward(self, x):
        if self.branch2 is not None:
            x = self.branch_fusion(x, self.branch2(x))
        return x

class Net(nn.Module):
    def __init__(self, basic_model):
        super(Net, self).__init__()
        self.blocks = nn.ModuleList([basic_model, basic_model, basic_model])

    def forward(self, x):
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        return x

class IdentityModel(nn.Module):
    def forward(self, a):
        return a

def my_model_function():
    return MyModel()

def GetInput():
    B, C, T, H, W = 1, 3, 16, 112, 112
    return torch.rand(B, C, T, H, W, dtype=torch.float32)

