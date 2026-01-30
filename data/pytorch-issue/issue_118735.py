import torch


def func():
    g = torch.Generator().manual_seed(42)
    t1 = torch.rand(1, generator=g)
    torch.manual_seed(42)
    t2 = torch.rand(1)
    return t1, t2


opt_func = torch.compile(func)

print(func())
print(opt_func())