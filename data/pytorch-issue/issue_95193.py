# torch.rand(2, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return torch.logit(tensor, eps=0.3)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32, requires_grad=True)

