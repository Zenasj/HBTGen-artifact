import torch.nn.functional as F

import torch
from transformers import FalconModel
from torch.nn import functional as F

torch.manual_seed(0)

a = 3
b = 4

q = torch.randn(size=(1, 1, a, b))
k = torch.randn(size=(1, 1, a, b))
v = torch.randn(size=(1, 1, a, b))

def check(q, k, v, device):

    q = q.to(device)
    k = k.to(device)
    v = v.to(device)

    neg_value = torch.finfo(q.dtype).min
    mask = [[neg_value, neg_value, neg_value], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    mask = torch.tensor([[mask]]).to(device)

    o = F.scaled_dot_product_attention(q, k, v, mask, 0.0, is_causal=False)
    print(o)

check(q, k, v, "cpu")
check(q, k, v, "cuda")