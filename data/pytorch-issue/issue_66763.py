import torch
import torch.nn as nn

class MatmulModel(torch.nn.Module):
    def forward(self,other):
        input = torch.randn(5, requires_grad=True).to(device)
        return torch.dot(input, other)

x = torch.randn(5, requires_grad=True)
test_mm = MatmulModel()
torch.onnx.export(test_mm, x.to(device), f = 'test.onnx')