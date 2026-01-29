# torch.rand(4096, 8192, dtype=torch.bfloat16, device="cuda")  # Non-contiguous input example
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.max(x)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")
    # Create non-contiguous tensor via transpose operations as in the original repro
    x = x.t().contiguous().t()  
    return x

