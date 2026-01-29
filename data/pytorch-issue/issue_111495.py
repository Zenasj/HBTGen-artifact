# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(5, 7)
    
    def forward(self, x):
        return self.fc0(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 5, dtype=torch.float32)

