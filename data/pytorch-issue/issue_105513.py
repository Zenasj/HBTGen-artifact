import torch

def fn(x):
    with torch.no_grad():
        assert x.max() < 5, f"invalid max {x.max()}"
        x = torch.sin(x)
    return x

x = torch.randn(4)
ref = fn(x)

opt_fn = torch.compile(fn, backend="aot_eager")
res = opt_fn(x)