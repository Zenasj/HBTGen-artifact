# torch.rand(1, 2, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lrn = nn.CrossMapLRN2d(size=1, alpha=0.1, beta=0.1, k=1)
    
    def forward(self, x):
        return self.lrn(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 1, 1, dtype=torch.float32)

