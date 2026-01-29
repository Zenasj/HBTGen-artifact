# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self, a, b):
        super(MyModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        tmp = self.net(x)
        return torch.where(tmp <= 0.5, 0.4, 1.0)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 100000),
            MyModule(100000, 5)
        )

    def forward(self, x):
        return self.net(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(20, 10, dtype=torch.float32)

