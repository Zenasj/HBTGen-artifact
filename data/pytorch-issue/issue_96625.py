# torch.rand(1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1, 2))  # Fixed bias tensor shape [1,2]

    def forward(self, b):
        c = torch.matmul(b, b)  # Shape [1,1]
        return torch.add(c, self.a)  # Broadcast [1,1] + [1,2]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1)  # Input tensor shape [1,1]

