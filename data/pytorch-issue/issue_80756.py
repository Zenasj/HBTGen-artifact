import torch.nn as nn

import torch
from torch import nn, jit
from typing import List
from torch.onnx import export


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)

    def forward(self, x):
        outputs = jit.annotate(List[torch.Tensor], [])
        for i in range(x.size(0)):
            outputs.append(self.conv(x[i].unsqueeze(0)))
        return torch.stack(outputs, 0).squeeze()


inputs = torch.rand((3, 1, 5, 5, 5))
model = Model()
with torch.no_grad():
    output = model(inputs)
model_script = jit.script(model)
export(model_script, inputs, 'script.onnx',
       opset_version=11, example_outputs=output)