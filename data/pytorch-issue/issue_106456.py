import torch

@torch.compile
def f(x):
    x.mul_(2)
    return x + 1


x = torch.ones(2, 2)
# take a non-contiguous slice of x
x_view = x[:, 0]
# f should mutate it
out = f(x_view)