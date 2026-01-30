import torch

with torch.no_grad():
    torch.embedding_renorm_(weight, input, max_norm, norm_type)