import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F


class toy_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 8, 3)
        self.weight = nn.Parameter(torch.rand(8, 3, 3, 3))


    def forward(self, x):
        self.weight = nn.Parameter(torch.rand(8, 6, 3, 3))
        self.conv2d.weight = self.weight
        self.conv2d.in_channels = 6

        x = self.conv2d(x)

        return x


x = torch.rand((1, 6, 64, 64))

model = toy_model()

torch.onnx.export(
    model,
    x,
    'toy.onnx',
    verbose=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
)