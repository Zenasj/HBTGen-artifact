import torch.nn as nn

import torch
from torch import nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()

    def forward(self, x):
        return x.sum(dim=(2, 3), keepdim=True)

model = Test()

x = torch.zeros((16, 3, 256, 256))

torch.onnx._export(model, x, "test.onnx", verbose=True)