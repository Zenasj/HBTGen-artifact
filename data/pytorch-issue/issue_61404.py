import torch.nn as nn

import torch
import onnx

FILENAME = 'model.onnx'


class Model(torch.nn.Module):
    def forward(self, x0):
        x = torch.clamp(x0, max=6)
        return x

torch.onnx.export(Model(), torch.randn(5, 5), FILENAME, opset_version=11)
model = onnx.load(FILENAME)
onnx.checker.check_model(model)
print(model)