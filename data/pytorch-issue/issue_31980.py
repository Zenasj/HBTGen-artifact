# torch.rand(N, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Method 1: repeat_interleave
        out1 = x.repeat_interleave(100, dim=-1)
        # Method 2: expand and flatten
        expanded = x.unsqueeze(-1).expand(-1, 100)
        out2 = expanded.flatten(-2, -1)
        # Return 1.0 if outputs are close, else 0.0
        return torch.tensor(torch.allclose(out1, out2, atol=1e-7), dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10000)

