import torch
from torch import nn

# torch.rand(10000, 256, dtype=torch.uint8)  # Inferred from issue's example input shape
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x  # Dummy model that passes input through

def my_model_function():
    return MyModel()

def GetInput():
    return torch.testing.make_tensor(
        (10_000, 256), 
        dtype=torch.uint8, 
        device="cpu", 
        high=256
    )

