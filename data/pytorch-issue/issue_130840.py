# torch.rand(32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
    
    def forward(self, x):
        return self.linear.forward(x)  # Explicit forward() call causing the issue

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, dtype=torch.float32, device='cuda')

