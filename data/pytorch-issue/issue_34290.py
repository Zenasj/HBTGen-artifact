import torch.nn as nn

import torch
class RoundLayer(torch.nn.Module):
    def forward(self, x):
        return torch.round(x)
torch.onnx.export(RoundLayer(), torch.rand(1), 'round.onnx')

torch.onnx.export(RoundLayer(), torch.rand(1), 'round.onnx', opset_version=11)