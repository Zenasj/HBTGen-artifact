# torch.rand(B, S, E, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # Matches input shape (B, S, E)
        )

    def forward(self, x):
        # Self-attention using scaled_dot_product_efficient_attention internally
        attn_output, _ = self.mha(x, x, x)
        return attn_output

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size=2, sequence length=10, embedding dim=64
    return torch.rand(2, 10, 64)

