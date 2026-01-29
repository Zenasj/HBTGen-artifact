# torch.rand(2, 2, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, inp_size, out_size, inner_size):
        super().__init__()
        self.Linear1 = nn.Linear(inp_size, inner_size)
        self.Linear2 = nn.Linear(inner_size, out_size)
        self.x = None  # Stores intermediate result for comparison

    def forward(self, inp):
        res = self.Linear1(inp)
        self.x = res  # Critical assignment causing CUDA graph issues
        return self.Linear2(res)

def my_model_function():
    # Initialize with parameters from the original test case
    return MyModel(inp_size=2, out_size=2, inner_size=4)

def GetInput():
    # Generate random input matching expected shape (batch=2, features=2)
    return torch.rand(2, 2, device='cuda', dtype=torch.float32)

