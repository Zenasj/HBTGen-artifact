import torch.nn as nn

'xxx' in dir(sdm)   # is True
sdm = torch.compile(sdm)          # Type is now OptimizedModule
'xxx' in dir(sdm)   # is NOW False
sdm.xxx     # Still is accessable

import torch
from torch.nn import Module

class AModel(Module):
    def __init__(self, xxx):
        self.xxx = xxx

model = AModel(7)
print(f"model.xxx = {model.xxx}")
model = torch.compile(model)
print(f"model.xxx = {model.xxx}")

@torch.compile
def boom():
    print(f"model.xxx = {model.xxx}")

boom()

import torch


class Model(torch.nn.Module):
    def init(self, channels):
        super(Model, self).init()
        self.channels = channels
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

n, c, h, w = 8, 12, 1, 1
x = torch.randn((n, c, h, w))

model = Model(c)
print(f"channels in {model}: {'channels' in dir(model)}")  # True
print(model.channels)

opt_model = torch.compile(model)
print(f"channels in {opt_model}: {'channels' in dir(opt_model)}")  # False
print(opt_model.channels)