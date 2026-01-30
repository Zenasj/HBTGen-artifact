import torch
import torch.func
import torch.nn as nn

x = torch.randn(3, device='cuda')
y = torch.randn(1, 3, device='cuda')

def fn(x, y):
    # output of `dropout` should be [B, 1, 3] (B=3).
    return x + nn.functional.dropout(y, 0.3).mean(1)

# Errors for `randomness == 'different` because `mean` expects atleast 2-d tensor.
o = torch.func.vmap(fn, in_dims=(0, None), randomness='different')(x, y)