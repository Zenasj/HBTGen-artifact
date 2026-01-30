import math

import torch

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def f(x):
    y = x.item()
    torch._check_is_size(y)
    r = torch.arange(y, dtype=torch.float32)
    torch._check(r.size(0) == y)
    return r

f(torch.tensor([300]))

length = math.ceil((end - start) / step)