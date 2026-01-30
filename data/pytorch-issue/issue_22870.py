import torch.nn as nn

import onnxruntime
from io import BytesIO
from torch.onnx import OperatorExportTypes

class Model(torch.nn.Module):
    def forward(self, x):
        return x.unbind(0)

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

torch.onnx.export(
    Model(), x, BytesIO(), verbose=True,
    operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
)

torch.onnx.symbolic.unbind = lambda g, x, d: x
torch.onnx.symbolic.listunpack = lambda g, x, d: x

class Model(torch.nn.Module):
    def forward(self, x):
        return [torch.squeeze(out, 0) for out in torch.split(x, [1,1,1], dim=0)]#x.unbind(0)

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

import torch
print("torch version:", torch.__version__)

class Model(torch.nn.Module):
    def forward(self, x):
        return x.unbind(0)

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
torch.onnx.export(Model(), x, "/dev/null", verbose=True)