import torch

@torch.compile()
def fn(x):
    tmp = x.ceil()
    x.add_(10)
    return tmp

a = torch.zeros((), dtype=torch.int64)
fn(a)  # tensor(10)