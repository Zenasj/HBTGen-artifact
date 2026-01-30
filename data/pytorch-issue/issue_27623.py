import torch
import torch.nn as nn

batch_size = 5
seq_len = 10
embed_dim = 24
value_dim = 12
num_heads = 1

query = torch.randn(seq_len, batch_size, embed_dim)
key = torch.randn(seq_len, batch_size, embed_dim)
value = torch.randn(seq_len, batch_size, value_dim)

mha = nn.MultiheadAttention(embed_dim, num_heads, vdim=value_dim)
attn_out, attn_out_weights = mha(query, key, value)