import torch

@torch.compile()
def f(x):
    y = x.item()
    torch.export.constrain_as_size(y)
    return torch.zeros(y)

f(torch.tensor([3]))