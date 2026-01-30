import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
torch.set_default_device('cuda')

q = torch.randn(8, 128, 1024, 128)
k = torch.randn(8, 128, 1024, 128)
v = torch.randn(8, 128, 1024, 128)

@torch.compile(mode="reduce-overhead")
def f(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

for _ in range(2):
    f(q, k, v)
with torch.profiler.profile() as prof:
    for _ in range(5):
        f(q, k, v)
prof.export_chrome_trace("/home/dberard/sdpa.json")