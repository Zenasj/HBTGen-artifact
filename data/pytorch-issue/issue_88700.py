# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Example input features: 10, output 5
        self.register_buffer('buffer', torch.ones(5))  # Example buffer

    def forward(self, x):
        x = self.linear(x)
        return x + self.buffer

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, dtype=torch.float32)

