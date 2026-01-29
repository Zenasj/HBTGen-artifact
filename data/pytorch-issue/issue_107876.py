# torch.rand(32, 100, 1000, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(1000, 1000)
        self.second_layer = nn.Linear(100, 100)
        # Initialize weights to 1.0 as per the issue's setup
        self.first_layer.weight.data.fill_(1.0)
        self.second_layer.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.functional.gelu(x)
        x = x.transpose(-1, -2)
        x = self.second_layer(x)
        x = x.transpose(-1, -2)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 100, 1000, dtype=torch.float32)

