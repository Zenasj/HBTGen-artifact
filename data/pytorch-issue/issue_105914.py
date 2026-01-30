import torch
a = torch.randn(64, dtype=torch.float16)

@torch.compile
def f(a, dim=-1):
    na = torch.linalg.vector_norm(a, dim=dim)
    eps = 1e-8
    return na.clamp_min(eps)

f(a)