import torch

def fn(x):
    x = torch.ops.aten.sigmoid.default(x)
    return torch.ops.aten.mean.dim(x, [-1, -2], True)

x = torch.randn((1, 8, 8, 8))
opt_fn = torch._dynamo.optimize("inductor")(fn)
opt_fn(x)

real_out = fn(x)
compiled_out = opt_fn(x)
tol = 0.0001
print(torch.allclose(real_out, compiled_out, atol=tol, rtol=tol))