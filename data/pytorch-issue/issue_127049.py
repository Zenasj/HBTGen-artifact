import torch

def f(x):
    try:
        x.add_(1.5)
    except:
        return x + 1
    return x + 2

opt_f = torch.compile(f, backend="eager")
inp = torch.ones(3, dtype=torch.int64)
opt_f(inp)