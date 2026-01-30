import torch

@torch.compile(backend="aot_eager")
def f(a):
    return list(a.unbind(0))

x = torch.ones(2, 2, requires_grad=True).clone()
y, z = f(x)
# autograd error
y.mul_(2)