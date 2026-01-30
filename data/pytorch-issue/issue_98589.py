import torch

@compile()
def f(x, y, z):
    x.index_put_((y,), z)
    return x

x = torch.randn(3, 2, 4, device="cuda")
y = torch.arange(2, device="cuda")
z = torch.zeros(4, 2, 4, device="cuda")
f(x, y, z)