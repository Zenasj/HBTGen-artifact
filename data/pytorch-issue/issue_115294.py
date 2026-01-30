import torch

def fn(x):
    with torch.enable_grad():
        out = x + 1
    return out
    

opt_fn = torch.compile(fn, backend="inductor")  # also, aot_eager, aot_eager_decomp_partition, but not eager
with torch.no_grad():
    res = fn(torch.zeros(10, requires_grad=True))
    opt_res = opt_fn(torch.zeros(10, requires_grad=True))

    assert res.requires_grad == opt_res.requires_grad  # True != False