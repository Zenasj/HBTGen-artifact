# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1))  # 1x1 parameter
        
    def forward(self, x):
        # Multiply input by the weight (fixes original forward's missing layer reference)
        return x * self.weight

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

