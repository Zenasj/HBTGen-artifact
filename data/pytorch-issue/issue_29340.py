import torch

def fn(x):
    print(torch.isfinite(x))


s = torch.jit.script(fn)
fn(torch.randn(2, 2))
s(torch.randn(2, 2))