import torch.nn as nn

import torch
from io import BytesIO

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.log1p(x)

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

torch.onnx.export(Model(), x, BytesIO())