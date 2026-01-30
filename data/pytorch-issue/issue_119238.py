import torch

def fn(x):
    return torch.sin(x + 1)

x = torch.randn((2, 2), device="cuda")

ref = fn(x)
print(ref)
opt_fn = torch.compile(fn, backend="eager")
res = opt_fn(x)
print(res)

logger.log