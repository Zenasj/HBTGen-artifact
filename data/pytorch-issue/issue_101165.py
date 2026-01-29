# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.seq = nn.Sequential(nn.Linear(2, 2))
    
    def forward(self, x):
        return self.seq(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2)

