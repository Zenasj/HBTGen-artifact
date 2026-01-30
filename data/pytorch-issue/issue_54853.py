import torch

def torch_diff(x, n, dim=-1):
    out = torch.diff(x, dim=dim)
    if n > 1:
        for _ in range(n - 1):
            out = torch.diff(out, dim=dim)
    return out