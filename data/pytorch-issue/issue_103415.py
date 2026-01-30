import torch
import torch._dynamo

import torch._inductor.config

torch._inductor.config.fallback_random = True
torch._inductor.config.cpp_wrapper = True

def fn(x):
    y = torch.randint(0, 10, (4, 4), dtype=torch.int32)
    return y + x

opt_fn = torch._dynamo.optimize("inductor")(fn)

x = torch.rand((4, 4))

torch.manual_seed(42)
ref = fn(x)
torch.manual_seed(42)
res = opt_fn(x)
print(torch.max(torch.abs(res-ref)))