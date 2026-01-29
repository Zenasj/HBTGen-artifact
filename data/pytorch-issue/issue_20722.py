# Three tensors of shape (30, 20, 1024) passed as a tuple (q, k, v)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16):
        super().__init__()
        self.mod = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, inputs):
        q, k, v = inputs
        return self.mod(q, k, v)

def my_model_function():
    return MyModel()

def GetInput():
    sl, bs, embed_dim = 30, 20, 1024
    q = torch.rand(sl, bs, embed_dim)
    k = torch.rand(sl, bs, embed_dim)
    v = torch.rand(sl, bs, embed_dim)
    return (q, k, v)

