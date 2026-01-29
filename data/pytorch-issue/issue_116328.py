# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Operation that triggered type checking discrepancy (int + Tensor)
        return x + 1  # Originally reported as returning "int" by mypy in older versions

def my_model_function():
    return MyModel()

def GetInput():
    # Match 4D input shape (B, C, H, W) with example dimensions from the issue (2,2) expanded to 4D
    return torch.rand(2, 2, 1, 1, dtype=torch.float32)

