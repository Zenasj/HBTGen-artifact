import torch

def fn(a, b, c, d):
    return (a + b) @ (c + d)

opt_fn = torch.compile(fn, backend="eager", dynamic=False)
opt_fn(torch.randn(10, 20), torch.randn(1, 20), torch.randn(20, 15), torch.randn(1, 15))
opt_fn(torch.randn(5, 2), torch.randn(1, 2), torch.randn(2, 4), torch.randn(1, 4))