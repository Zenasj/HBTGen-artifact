import torch
import torch._dynamo
import logging

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True

def fn(x, y, z):
    return torch.index_put(x, [y], z)

x = torch.rand([8, 2])
y = torch.rand([8]) > 0.5
z = torch.rand([])
opt_fn = torch._dynamo.optimize("inductor")(fn)
print(fn(x, y, z))
print(opt_fn(x, y, z))

where(cond, a, b)

index_put_fallback