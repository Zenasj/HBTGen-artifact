import torch

d = {
    torch.float16: torch.float32,
}

@torch.compile
def f():
    return torch.randn(3, dtype=d[torch.float16])

f()