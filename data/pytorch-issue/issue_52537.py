# torch.rand(5, 2, 8, dtype=torch.float)  # Input shape: (seq_len=5, batch=2, embed_dim=8)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, embed_dim=8, num_heads=2, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias)

    def forward(self, x):
        # Self-attention using the same tensor for Q/K/V
        # Returns output tensor of shape (seq_len, batch, embed_dim)
        return self.mha(x, x, x)[0]

def my_model_function():
    # Returns a model with bias disabled for both in/out projections (matches test case)
    return MyModel(bias=False)

def GetInput():
    # Generates a random input tensor matching the expected MultiheadAttention input shape
    return torch.rand(5, 2, 8, dtype=torch.float)

