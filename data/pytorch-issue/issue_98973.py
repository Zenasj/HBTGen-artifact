# torch.rand(2, 3, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, input):
        permute = torch.tensor([0, 2, 1])
        x = input.permute(*permute)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3, 4)

