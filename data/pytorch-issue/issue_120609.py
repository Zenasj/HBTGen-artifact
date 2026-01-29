# torch.rand(B, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(100, 50)
        
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 100, dtype=torch.float32)

