import torch.nn as nn

import torch


class Zeros(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.zeros(())
        y += x
        return y


x = torch.tensor(42.)
torch.onnx.export(Zeros(), x, 'zeros.onnx')

torch.onnx._export(
    Zeros(), x, 'zeros.onnx',
    enable_onnx_checker=False,
    onnx_shape_inference=False,
)
print(onnx.load('zeros.onnx'))