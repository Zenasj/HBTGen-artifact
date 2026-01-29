# torch.rand(B, S, E, dtype=torch.float32)  # e.g., B=1, S=5, E=10
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=10,
            num_heads=2,  # Even number required to trigger the bug in newer versions
            bias=False,
            batch_first=True
        )
    
    def forward(self, x):
        # Forward pass uses x as query, key, and value to replicate the original issue's call pattern
        return self.attention(x, x, x)[0]  # Return only the output tensor (ignores attention weights)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 5, 10)  # Matches the input shape (B, S, E) with batch_first=True

