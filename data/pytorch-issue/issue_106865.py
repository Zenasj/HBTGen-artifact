import torch

@torch.compile()
def fn(x, y):
    return torch.mm(x, y)

inps = [torch.rand([0, 30]), torch.rand([30, 40])]
inps = [x.to(device="cuda") for x in inps]
out = fn(*inps)