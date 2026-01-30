import torch
import torch.nn as nn

def torch_impl_layernorm(x):
    m = nn.LayerNorm(x.shape[-1], elementwise_affine=False)
    return m(x)


def my_impl_layernorm(x):
    v = x.var(dim=-1, unbiased=False, keepdim=True)
    e = x.mean(dim=-1, keepdim=True)
    v = v + 1e-5  # same epsilon as torch default
    v = v.sqrt()
    out = (x - e) / v
    return out

H = 2000
W = 100
n_channel = 32
a = torch.arange(H * W * n_channel).reshape((H, W, n_channel)).float()
b = my_impl_layernorm(a)
c = torch_impl_layernorm(a)
print((c - b).abs().mean()) #  output: tensor(1996.2400)
print(c)