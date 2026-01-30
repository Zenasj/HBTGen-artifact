import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def f(x):
    y = x.item()
    return torch.ones(y).sum()

f(torch.tensor(5))