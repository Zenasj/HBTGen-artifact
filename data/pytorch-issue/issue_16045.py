# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(2, 1)  # Matches the original model structure
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    sample_size = 100  # Inferred from the original code example
    return torch.randn((sample_size, 2), dtype=torch.float32)

