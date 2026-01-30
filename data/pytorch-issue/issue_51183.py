import torch.nn as nn

import torch

# create a simple module
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x):
        return x.T

dummy_input = torch.randn(1, 300)
torch.onnx.export(MyModule(), dummy_input, "aten_transpose_issue.onnx", opset_version=13, verbose=True)