# torch.rand(B), torch.rand(B)  # dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    B = torch.randint(1, 10, (1,)).item()  # Random batch size between 1-9
    return (torch.rand(B), torch.rand(B))

