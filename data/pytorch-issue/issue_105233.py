# torch.rand(B, S, dtype=torch.float)  # Assuming input is 2D tensor for sequence data
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # Register buffer with persistent=False to replicate the issue
        self.register_buffer("bias", torch.ones(1), persistent=False)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.ModuleList([Block()])  # Matches "h.0" submodule path

    def forward(self, x):
        # Example computation using the buffer (to make the model usable)
        return x + self.h[0].attn.bias

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (batch, sequence_length)
    return torch.rand(2, 5)  # B=2, S=5 as an example

