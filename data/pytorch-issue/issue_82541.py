# torch.rand(1, 0, dtype=torch.float32), torch.rand(1, 0, dtype=torch.float32)  # Tuple of two tensors with shape (1,0)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return torch.true_divide(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(1, 0, dtype=torch.float32, requires_grad=True)
    b = torch.rand(1, 0, dtype=torch.float32, requires_grad=True)
    return (a, b)

