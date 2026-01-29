# torch.rand(B, S, E, dtype=torch.float32)  # B=batch_size, S=sequence_length, E=embedding_dim
import torch
import torch.nn as nn

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        tgt_len, bsz, embed_dim = query.size()
        src_len, _, _ = key.size()
        assert embed_dim == self.embed_dim, f"Expected {self.embed_dim}, got {embed_dim}"

        qkv = self.in_proj(query)
        q, k, v = torch.split(qkv, self.embed_dim, dim=-1)

        # Fix: use src_len/tgt_len as symbolic dimensions instead of tensor.shape[0]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Simplified attention computation (for demonstration)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        return self.out_proj(attn_output), None  # Dummy attention weights

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.nhead = 8
        self.attention = CustomMultiheadAttention(self.d_model, self.nhead)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor matching (batch_size, seq_len, embedding_dim)
    return torch.rand(1, 30, 512, dtype=torch.float32)

