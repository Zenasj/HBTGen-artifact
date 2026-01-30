import torch
import torch._dynamo
import torch._inductor

def fn(a, s):
    x = torch.rand([s, s, s, s], device="cuda")
    return a, x.sum()

opt_fn = torch._dynamo.optimize("inductor")(fn)
opt_fn(torch.rand([100, 100], device="cuda"), 1000)

1000

100