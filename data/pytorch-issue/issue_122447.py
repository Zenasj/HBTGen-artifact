import torch
from torch import nn

# torch.rand(B, C, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 20)
        self.net2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 10, dtype=torch.float32)

