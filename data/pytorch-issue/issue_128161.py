import torch

def fn(n):
    x = torch.full((1,), n >= 1024, dtype=torch.bool, device="cuda")
    return x + 1

opt_fn = torch.compile(fn, backend="inductor", dynamic=True)
print(opt_fn(1024))