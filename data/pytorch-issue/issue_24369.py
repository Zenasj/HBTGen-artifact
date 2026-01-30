import torch
import torch.nn as nn

class Mod(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, downsample=None):
        super(Mod, self).__init__()
        self.downsample = nn.Sequential(downsample)

    def forward(self, input):
        if self.downsample is not None:
            return self.downsample(forward)
        return input

torch.jit.script(Mod(nn.Linear(10, 20)))