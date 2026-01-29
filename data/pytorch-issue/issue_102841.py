# torch.rand(4, dtype=torch.float32)  # x shape (4,), b is a BoolTensor of shape (4,), y shape (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, b, y = inputs
        x = x.clone()
        x[b] = y
        return x

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(4, requires_grad=True)
    b = torch.tensor([True, False, True, False])
    y = torch.randn(2, requires_grad=True)
    return (x, b, y)

