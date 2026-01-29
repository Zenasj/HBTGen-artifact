# torch.rand(5, 8, 8, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bmm_weight = nn.Parameter(torch.ones(8, 8, dtype=torch.float32))

    def forward(self, x):
        return (self.bmm_weight * x).sum(dim=(1, 2))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 8, 8, dtype=torch.float32)

