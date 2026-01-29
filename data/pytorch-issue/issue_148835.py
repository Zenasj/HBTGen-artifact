# torch.rand(1, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.relu_(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 10)

