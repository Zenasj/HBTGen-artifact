# torch.rand(2, 3, dtype=torch.float32)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        x = torch.nn.functional.tanh(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)

