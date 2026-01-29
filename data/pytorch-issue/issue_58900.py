# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        # Example in-place operation (add_)
        x = self.linear(x)
        x.add_(2)  # In-place modification to trigger grad check
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input with requires_grad=True to enable grad checks
    return torch.rand(2, 10, requires_grad=True)

