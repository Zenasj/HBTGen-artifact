@torch.compile(backend="eager", dynamic=True)
def f(t):
    return t._base + 1

x_a = torch.randn(4, 4, requires_grad=True)
x = TwoTensor(x_a, x_a.clone())
out = f(x[3])

import torch
from torch.testing._internal.two_tensor import TwoTensor

def f(x):
    return x * 2

f_compiled = torch.compile(f, backend="aot_eager", dynamic=True)

a, b = (torch.randn(4, 15) for _ in range(2))
t = TwoTensor(a, b)

t_view = t.view(-1)
out_ref = f(t_view)
out_test = f_compiled(t_view)