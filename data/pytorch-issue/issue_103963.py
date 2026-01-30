import torch.nn as nn

import torch
import torch.nn.functional as F

torch.manual_seed(13)

q = torch.rand(1, 1, 8, 8, device='cuda')
k = torch.rand(1, 1, 8, 8, device='cuda')
v = torch.rand(1, 1, 8, 8, device='cuda')

attn_mask = torch.rand(1, 1, 8, 8, device='cuda')
# make only attn_mask[..., :4, :4] nonzero
attn_mask[..., 4:, :] = 0
attn_mask[..., :, 4:] = 0
attn_mask[..., 4:, 4:] = 0

def run(attn_mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask)

print(run(attn_mask))
print()
print(run(attn_mask.to(torch.bool).to(torch.float)))
print()
print(run(attn_mask.to(torch.bool)))

In [14]: x = torch.tensor([-float('inf')] * 3)

In [15]: x.softmax(-1)
Out[15]: tensor([nan, nan, nan])

In [16]: x[0] = 1

In [17]: x.softmax(-1)
Out[17]: tensor([1., 0., 0.])