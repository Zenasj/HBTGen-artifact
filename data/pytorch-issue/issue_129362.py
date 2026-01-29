# torch.rand(B, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 5)

    def forward(self, x):
        return self.lin(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 5, dtype=torch.float32).cuda()

