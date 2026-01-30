import torch.nn as nn

import torch
from torch import nn

class TorchAll(nn.Module):
    def forward(self, tensor):
        tensor = torch.all(tensor, dim=1)
        return tensor

X = torch.ones((3, 300, 300), dtype=torch.int32)

torch.onnx.export(
    TorchAll(),
    (X), # Dummy input for shape
    "torch_all_model.onnx",
    opset_version=12,
    do_constant_folding=True,
)