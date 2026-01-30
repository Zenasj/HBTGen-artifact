import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1))
        return self.conv(x)

torch.onnx.export(Model(), torch.randn(1, 2, 9, 9), 'm.onnx')