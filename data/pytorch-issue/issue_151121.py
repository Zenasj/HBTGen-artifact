import torch.nn as nn

py
import torch


class BfloatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))

    def forward(self, x):
        return x * torch.tensor(1.0, dtype=torch.bfloat16) * self.param


input = torch.randn(1, 10, dtype=torch.bfloat16)
model = BfloatModel()
onnx_program = torch.onnx.export(model, (input,), dynamo=True, optimize=False, verify=True)