# torch.rand(1, dtype=torch.bool), torch.rand(1, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        mask, tensor_b = inputs
        return torch.ops.aten.where.ScalarSelf(mask, True, tensor_b)

def my_model_function():
    return MyModel()

def GetInput():
    mask = torch.tensor([False], dtype=torch.bool)
    tensor_b = torch.tensor([False], dtype=torch.bool)
    return (mask, tensor_b)

