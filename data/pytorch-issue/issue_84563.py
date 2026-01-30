import torch
import torch.nn as nn

model = torch.nn.LayerNorm(256)
x = torch.rand(950, 1, 256)
torch.onnx.export(
    model,
    x,
    'tmp.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=11)