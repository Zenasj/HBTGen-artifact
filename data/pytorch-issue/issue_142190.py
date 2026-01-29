# torch.rand(8, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        # Use mm instead of matmul to align with DTensor-compatible ops (as per issue's solution suggestion)
        return torch.mm(x, self.weight)

def my_model_function():
    dim = 128  # Matches the 'dim' variable in the original repro
    return MyModel(dim)

def GetInput():
    # Generate input tensor matching the repro's x.shape (8, 128)
    return torch.randn(8, 128)

