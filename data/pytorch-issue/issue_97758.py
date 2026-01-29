# torch.rand(1, dtype=torch.float16), torch.rand(1, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        t = torch.linalg.vector_norm(x)
        return torch.remainder(y, t)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1, dtype=torch.float16), torch.rand(1, dtype=torch.float16))

