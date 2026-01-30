import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

def g(x, y):
    y = x.item()
    if y < 3:
        return x + 2
    else:
        return x + 3

@torch.compile()
def f(x, y):
    y = y * y
    return g(x, y)

f(torch.tensor(4), torch.randn(4))