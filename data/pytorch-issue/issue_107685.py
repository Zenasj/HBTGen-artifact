import torch.nn as nn

import torch
from torch import nn


class Model(nn.Module):
    def forward(self, x):
        x = nn.functional.glu(x, dim=1)
        return x

model = Model()
model.eval()
x = torch.rand(1024, 512)

torch.onnx.export(
    model, (x,),
    "model.onnx",
    verbose=False,
    opset_version=18,
)