import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def f(x):
    y = x.item()
    z = torch.randn(y, 2048)
    r = torch.cat([z, torch.randn(2, 2048)])
    return r[:, 0:152]

f(torch.tensor(4))