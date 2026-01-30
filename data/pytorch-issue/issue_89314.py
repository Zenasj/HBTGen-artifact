import torch

def fn(a):
    return a.repeat_interleave(14, dim=0).repeat_interleave(14, dim=1)

x = torch.ones(14, 14).to(dtype=torch.int64)
opt_fn = torch._dynamo.optimize("eager")(fn)
opt_fn(x)