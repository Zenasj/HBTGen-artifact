import torch
import torch.nn as nn

class Module(nn.Module):
    def forward(self, x):
        out = (x == 0).all(1)
        return out

x = torch.zeros((1, 64), dtype=torch.long)
onnx_path = 'model.onnx'
model = Module()
torch.onnx.export(model, x, onnx_path)

In [16]: m = Module()

In [17]: m(x)
Out[17]: tensor([True])