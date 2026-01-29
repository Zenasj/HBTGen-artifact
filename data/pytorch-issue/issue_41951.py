# torch.rand(10, 2, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)

    def forward(self, x):
        # MultiheadAttention expects (seq_len, batch, embed_dim)
        return self.mha(x, x, x)[0]

def my_model_function():
    return MyModel()

def GetInput():
    # Example input shape matching embed_dim=64
    return torch.rand(10, 2, 64, dtype=torch.float32)

