import torch
from torch import nn

# torch.rand(1, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, a):
        return a + self.b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

