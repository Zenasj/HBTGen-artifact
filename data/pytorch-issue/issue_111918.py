import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile(backend="eager", dynamic=True)
def fn(x):
    x = x + 1
    y = x.item()
    if y > 2:
        return x * 2
    else:
        return x * 3

x = torch.tensor([3.0])
fn(x)