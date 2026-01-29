# torch.rand(B, 512, 16, 16, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(512, 1024))  # Matrix multiplication weight
        
    def forward(self, x):
        # Flatten spatial dimensions into batch for matrix multiplication
        x = x.view(-1, x.size(1))  
        return torch.mm(x, self.weight)  # Trigger matrix multiplication with pad_mm pass

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    return torch.rand(B, 512, 16, 16, dtype=torch.float32)

