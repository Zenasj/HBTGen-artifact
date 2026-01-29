# torch.rand(1, 3, 416, 416, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
from collections import OrderedDict

class DarknetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bnorm=True, leaky=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bnorm = nn.BatchNorm2d(out_channels, eps=1e-3) if bnorm else None
        self.leaky = nn.LeakyReLU(0.1) if leaky else None

    def forward(self, x):
        x = self.conv(x)
        if self.bnorm is not None:
            x = self.bnorm(x)
        if self.leaky is not None:
            x = self.leaky(x)
        return x

class DarknetBlock(nn.Module):
    def __init__(self, layers, skip=True):
        super().__init__()
        self.skip = skip
        self.block = OrderedDict()
        for i in range(len(layers)):
            self.block[layers[i]['id']] = DarknetLayer(
                layers[i]['in_channels'], 
                layers[i]['out_channels'], 
                layers[i]['kernel_size'],
                layers[i]['stride'], 
                layers[i]['padding'], 
                layers[i]['bnorm'],
                layers[i]['leaky']
            )
        self.block = nn.ModuleDict(self.block)

    def forward(self, x):
        count = 0
        for key in self.block:
            print(key)
            if count == (len(self.block) - 2) and self.skip:
                skip_connection = x
            count += 1
            x = self.block[key](x)
        return x + skip_connection if self.skip else x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.block0_4 = DarknetBlock([
            {'id': 'layer_0', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_1', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_2', 'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_3', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.block5_8 = DarknetBlock([
            {'id': 'layer_5', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_6', 'in_channels': 128, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_7', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.block9_11 = DarknetBlock([
            {'id': 'layer_9', 'in_channels': 128, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_10', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])

    def forward(self, x):
        x = self.block0_4(x)
        x = self.block5_8(x)
        x = self.block9_11(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 416, 416, dtype=torch.float32)

