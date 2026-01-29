# torch.rand(2, 2, 3, 3, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        v1_0 = torch.movedim(x, source=1, destination=2)
        v4_0 = x.add_(1)
        v0_0 = torch.cat([v4_0, v4_0], dim=2)
        return [v1_0, v0_0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([2, 2, 3, 3, 2], dtype=torch.float32)

