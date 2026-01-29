# torch.rand(B, 8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 32, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 32, bias=False)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 8, dtype=torch.float32)

