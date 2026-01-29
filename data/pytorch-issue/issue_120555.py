# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10, 2)  # Matches the example's model structure
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Example batch size, arbitrary but consistent
    return torch.rand(B, 10, dtype=torch.float32)

