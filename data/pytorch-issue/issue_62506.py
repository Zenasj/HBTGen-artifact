# torch.rand(2, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        elements = torch.tensor([2, 3], device=x.device)
        return torch.isin(x, elements)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32, requires_grad=True)

