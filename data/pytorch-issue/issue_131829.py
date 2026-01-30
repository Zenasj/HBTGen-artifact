import torch

# pip install rotary-embedding-torch
from rotary_embedding_torch import RotaryEmbedding

embed = RotaryEmbedding(dim=12)
inputs = torch.rand(3, 8, 10, 12)
torch.export.export(embed, (inputs,))