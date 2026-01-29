# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (N, L) where N is batch size and L is sequence length

import torch.nn as nn
import torch.nn.functional as F
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(128, embedding_dim=128)
        self.cross_attn = MultiheadAttention(128, 8)

    def forward(self, tokens):
        embed = self.tok_emb(tokens)
        z = torch.randn(embed.shape[0], 4, embed.shape[2], device=embed.device, dtype=embed.dtype)
        logits = self.cross_attn(embed, z, z)
        return logits.mean()

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self.embed_size = embed_dim
        self.num_heads = heads
        self.head_dim = embed_dim // heads

        assert self.head_dim * heads == embed_dim, "Embedding size needs to be divisible by heads"

        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.k_proj = nn.Linear(self.embed_size, self.embed_size)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]

        q = query.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = key.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = value.view(N, -1, self.num_heads, self.head_dim).swapaxes(1, 2)

        # Ensure all tensors have the same dtype
        if q.dtype != k.dtype or q.dtype != v.dtype:
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.swapaxes(1, 2).reshape(N, -1, self.embed_size)
        return attn

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel().to(device="cuda", dtype=torch.bfloat16)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 128, (512, 128), device=torch.device("cuda"))

