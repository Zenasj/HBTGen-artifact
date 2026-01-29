# torch.rand(B, 3, H, W, dtype=torch.float32)

import torch
import torch.nn as nn
import math

def get_conv(in_channels, out_channels, kernel_size=3, actn=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)]
    if actn:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ResSequential(nn.Module):
    def __init__(self, layers, mult):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.mult = mult
    
    def forward(self, x):
        return x + self.layers(x) * self.mult

def res_block(num_features):
    layers = [
        get_conv(num_features, num_features),
        get_conv(num_features, num_features, actn=False)
    ]
    return ResSequential(layers, 0.1)

def upsample(in_channels, out_channels, scale):
    layers = []
    for _ in range(int(math.log(scale, 2))):
        layers += [get_conv(in_channels, out_channels * 4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)

class MyModel(nn.Module):
    def __init__(self, scale, nf=64):
        super().__init__()
        layers = [
            get_conv(3, nf),
            *[res_block(nf) for _ in range(8)],
            upsample(nf, nf, scale),
            nn.BatchNorm2d(nf),
            get_conv(nf, 3, actn=False),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel(scale=2)

def GetInput():
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

