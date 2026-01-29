# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)  # Matches parameter dimensionality from issue examples
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size placeholder (issue examples use single-element batches implicitly)
    return torch.rand(B, 2, dtype=torch.float32)

