import torch
from torch.nested._internal.nested_tensor import jagged_from_list

def fn(x, y):
    z = x.sin()
    y.sin_()
    return z.cos(), y.cos()

fn_c = torch.compile(fn, backend="inductor")

values = [torch.rand((i, 8), requires_grad=True) for i in range(1, 6)]
values_copy = [x.detach().clone().requires_grad_(True) for x in values]

nt, offsets = jagged_from_list(values, None)
nt_copy, offsets = jagged_from_list(values_copy, offsets)
y = torch.rand((4, 8))
y_copy = y.clone()

ret = fn_c(nt, y)[0]
ref = fn(nt_copy, y_copy)[0]