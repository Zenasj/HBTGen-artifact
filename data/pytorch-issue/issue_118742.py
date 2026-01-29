# torch.rand(10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 30)
        self.lin = nn.utils.weight_norm(self.lin, dim=0)
    
    def forward(self, x):
        return self.lin(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10)

