import torch

@torch.jit.script
def fn(x):
    return x*x + x -3

x=torch.randn(4, device="cuda")
for _ in range(10):
    fn(x)