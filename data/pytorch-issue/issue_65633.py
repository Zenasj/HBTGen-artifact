# torch.rand(2, 3, dtype=torch.float32), torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        mat, vec = inputs
        return torch.mv(mat, vec)

def my_model_function():
    return MyModel()

def GetInput():
    mat = torch.rand(2, 3, dtype=torch.float32)
    vec = torch.rand(3, dtype=torch.float32)
    return (mat, vec)

