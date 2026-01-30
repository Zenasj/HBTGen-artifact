import torch
import torch._dynamo

@torch.compile(backend="eager", fullgraph=True)
def fn(x):
    return torch.functional.split(x, 0)

fn(torch.empty((0,)))