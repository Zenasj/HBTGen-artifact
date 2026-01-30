import torch.nn as nn

import numpy as np
import torch

class MyNumpyModel(torch.nn.Module):
    def __init__(self):
        super(MyNumpyModel, self).__init__()

    def forward(self, input):
        return input.numpy()

with torch._subclasses.FakeTensorMode():
    model = MyNumpyModel()
    _ = torch.export.export(model, args=(torch.randn(1000),))