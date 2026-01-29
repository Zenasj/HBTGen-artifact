# torch.rand(B, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.register_buffer("buf", torch.randn(4,), persistent=False)
        self.weight = nn.Parameter(torch.randn(4, 4))
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 4, dtype=torch.float32)  # Batch size 2, 4 features

