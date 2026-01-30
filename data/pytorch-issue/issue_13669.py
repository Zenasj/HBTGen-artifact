import torch
import torch.autograd as autograd

def fn(z):
    return (z.exp() - 0).sum() + (z.exp() - torch.zeros(1000)).sum()

z = torch.tensor(1., requires_grad=True)
fn_jit = torch.jit.trace(fn, (z,))
print(fn(z))
print(fn_jit(z))
print(autograd.grad(fn(z), (z,)))
print(autograd.grad(fn_jit(z), (z,)))