import torch

def f(m, a, b):
    try:
        res = torch.addr(m, a, b, beta=True, alpha=2)
    except:
        return m + 1
    return res

opt_f = torch.compile(f, backend="eager")
m = torch.randn(8, 8)
a = torch.rand(8)
b = torch.rand(8)
opt_f(m, a, b)