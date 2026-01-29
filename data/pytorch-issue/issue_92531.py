# torch.rand(B, C, L, dtype=torch.float32)  # Input shape is (20, 16, 50) as per the example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(3, stride=2)
    
    def forward(self, x):
        # Transpose dimensions 1 and 2 before applying MaxPool1d
        x = x.transpose(1, 2)
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the original issue's dimensions (20,16,50)
    return torch.rand(20, 16, 50, dtype=torch.float32)

