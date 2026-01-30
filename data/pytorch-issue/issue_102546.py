import torch

def fn(x):
    return x.mean(0)

x = torch.randn((), device="cuda")
print("eager ", fn(x))
opt = torch.compile(fn)
print("compiled ", opt(x))