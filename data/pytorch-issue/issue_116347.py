# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class CustomModule(nn.Module):
    def forward(self, x):
        return x * 2  # Example of Python module operation

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom = CustomModule()  # Python-wrapped module
        self.linear = nn.Linear(3, 5)  # Input features: 3

    def forward(self, x):
        x = self.custom(x)
        x = self.linear(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size
    return torch.rand(B, 3, dtype=torch.float32)

