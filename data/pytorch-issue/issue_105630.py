import torch
import torch.nn as nn

p = torch.nn.MultiheadAttention(embed_dim=0, num_heads=100)

p = torch.nn.MultiheadAttention(embed_dim=100, num_heads=0)

p = torch.nn.MultiheadAttention(embed_dim=-1, num_heads=1)