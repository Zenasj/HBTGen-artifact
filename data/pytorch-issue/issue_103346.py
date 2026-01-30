import torch

def f(x):
    a = x.item()
    constrain_as_size(a, 4, 7)
    return torch.empty((a, 4))

inp = torch.tensor([5])
ep = torch._export.export(f, (inp,))

...
inp = torch.tensor([10])
ep = torch._export.export(f, (inp,)) # immediately raise error