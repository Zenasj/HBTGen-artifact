# torch.rand(S, N, E, dtype=torch.float32)  # S=10 (sequence length), N=1 (batch), E=512 (embed_dim)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, bias=False)
    
    def forward(self, x):
        # Self-attention using the same input for query/key/value
        # Returns only the output tensor (ignores attention weights)
        return self.attention(x, x, x)[0]

def my_model_function():
    # Returns the problematic model instance (with bias=False)
    return MyModel()

def GetInput():
    # Generate random input matching (S, N, E) shape expected by MultiheadAttention
    return torch.rand(10, 1, 512, dtype=torch.float32)

