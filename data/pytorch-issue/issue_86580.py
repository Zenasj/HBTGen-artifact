# torch.rand(B, C, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, C=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(C, eps=1e-8)
        
    def forward(self, x):
        return self.layer_norm(x)

def my_model_function():
    model = MyModel()
    return model.half()  # Matches original issue's fp16 model initialization

def GetInput():
    B = 8  # Matches issue's minimal example batch size
    C = 4   # Matches issue's channel dimension
    return torch.rand(B, C, dtype=torch.float16)

