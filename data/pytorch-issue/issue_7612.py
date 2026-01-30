import torch
import torch.nn as nn

import torch.onnx
import onnx

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x.view(-1, 784)

model = Net()
x = torch.randn(64, 1, 28, 28, requires_grad=True)
torch_out = torch.onnx._export(model, x, "view.proto", export_params=True, verbose=True)
model = onnx.load("view.proto")
onnx.checker.check_model(model)