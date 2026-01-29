# torch.rand(B, 1, 256, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer_norm = nn.LayerNorm(256)  # Matches the issue's LayerNorm(256) configuration
    
    def forward(self, x):
        return self.layer_norm(x)

def my_model_function():
    return MyModel()  # Returns the fixed model instance

def GetInput():
    B = 950  # Matches the issue's input batch size
    return torch.rand(B, 1, 256, dtype=torch.float32)  # Reproduces the original input shape

