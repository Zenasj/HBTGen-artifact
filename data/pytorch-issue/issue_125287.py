import torch

@torch.compile(backend='aot_eager')
def f(x):
    y = x.sin().sin()
    torch.autograd.backward([y], [torch.ones_like(y)])

x = torch.ones(4, requires_grad=True)
f(x)
print(x_ref.grad)