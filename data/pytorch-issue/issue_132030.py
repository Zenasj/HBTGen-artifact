import torch.nn as nn

import torch
import torch.onnx
class MyModel(torch.nn.Module):
    def forward(self, x):
        return x * torch.onnx.is_in_onnx_export()
torch.manual_seed(0)
model = MyModel().cuda().eval()
x = torch.tensor([[0.1, 0.2]], device='cuda', dtype=torch.float32)
torch.onnx.export(model, (x, ), "test_is_in_onnx_export.onnx")

import torch
import torch.onnx
class MyModel(torch.nn.Module):
    def forward(self, x):
        return x * torch.onnx.is_in_onnx_export()
torch.manual_seed(0)
model = MyModel().cuda().eval()
x = torch.tensor([[0.1, 0.2]], device='cuda', dtype=torch.float32)
y = model(x)
print(y) # tensor([[0., 0.]], device='cuda:0')

In [2]: torch.tensor(torch.onnx.is_in_onnx_export(), dtype=torch.float32)
Out[2]: tensor(0.)