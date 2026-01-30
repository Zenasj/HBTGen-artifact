import torch.nn as nn

import onnx
import torch


class Model(torch.nn.Module):
    def forward(self, c, x):
        # OK if you specify dtype explicitly.
        # return torch.where(x > 0, x, torch.zeros(size=(), dtype=x.dtype))
        return torch.where(c, x, 0.0)


c = torch.zeros(8, dtype=torch.bool)
x = torch.zeros(8)
torch.onnx.export(Model(), (c, x), 'where_bug.onnx')

model = onnx.load("where_bug.onnx")
print(model.graph)