import torch

@torch.compile(dynamic=True)
def f(*args):
    for a in args:
        a.add_(1)
    return args[0]

x = torch.ones(1000)
args = x.split(10)
out = f(*args)