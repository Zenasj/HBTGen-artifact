import torch.nn as nn

py
import torch
from torch import nn


class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.linear = nn.Bilinear(in1_features=0, in2_features=0, out_features=0)

    def forward(self, x):
        # 1st block
        x = self.conv(x)
        x = self.linear(x)

        return x


if __name__ == '__main__':
    net = lenet()