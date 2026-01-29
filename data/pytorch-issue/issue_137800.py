# torch.rand(B, C, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, block_size=32, type_size=34, dtype=torch.float32):
        super().__init__()
        self.block_size = block_size
        self.type_size = type_size
        self.dtype = dtype

    def forward(self, blocks):
        # Split into first 2 columns (d) and the rest (x)
        d, x = torch.split(blocks, [2, blocks.shape[1] - 2], dim=1)
        # Critical step: reinterpret d as float32 (from float16) causing MPS buffer error
        d = d.view(torch.float32).to(self.dtype)  # Triggers buffer size issue on MPS
        x = x.view(torch.int8)
        # Cast x to d's dtype for multiplication
        qb = d * x.to(d.dtype)
        return qb

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces the failing scenario with MPS buffer error
    return torch.rand(16, 16, dtype=torch.float16, device='mps' if torch.backends.mps.is_available() else 'cpu')

