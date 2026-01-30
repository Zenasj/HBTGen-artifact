import torch
import torch.func
import torch.nn as nn

x = torch.randn(3, device='cuda')
y = torch.randn(1, 3, device='cuda')

def fn(x, y):
    # previously output of dropout used to be incorrect [B, 3] (B=1) and thus `mean(1)` used to fail
    # post the fix output of dropout is [B, 1, 3] and `mean(1)` works.
    return x + nn.functional.dropout(y, 0.3).mean(1)


o = torch.func.vmap(fn, in_dims=(0, None), randomness='different')(x, y)