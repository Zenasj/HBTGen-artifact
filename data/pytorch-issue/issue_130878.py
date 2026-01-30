import torch

def f(x):
    return torch.sin(x)

optimized = torch.compile(f, mode="reduce-overhead")
print(optimized(torch.randn(10, device="cuda")))