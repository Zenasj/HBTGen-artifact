import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        x = self.bn(x)
        return x