import torch.nn as nn

import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def f(q, k, v):
    is_causal = q.shape[2] > 1
    return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

query = torch.randn(1, 32, 17, 128)
key, value = query, query
output = f(query, key, value)
print(output.shape)