import torch


@torch.compile(backend="aot_eager")
def f(x):
    x.resize_(6)
    x.mul_(2)
    return x

def g(x):
    x.resize_(6)
    x.mul_(2)
    return x

a = torch.ones(4)
b = torch.ones(4)
out = f(a)
out2 = g(b)
print(a.shape)
print(b.shape)
print(torch.allclose(a, b))
print(torch.allclose(out, out2))