# torch.rand(1, dtype=torch.int32), torch.rand(0, 0, dtype=torch.int32), torch.tensor([], dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        input, mat, vec = inputs
        return torch.addmv(input, mat, vec)

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.tensor([2], dtype=torch.int32)
    mat = torch.empty((0, 0), dtype=torch.int32)
    vec = torch.tensor([], dtype=torch.int32)
    return (input, mat, vec)

