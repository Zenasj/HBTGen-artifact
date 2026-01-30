import torch
from torch.nested._internal.nested_tensor import jagged_from_list

def get_jagged_tensor(nested_size, offsets):
    D = nested_size[1]
    out = []
    for s in nested_size[0]:
        out.append(torch.randn(s, D, requires_grad=False, dtype=torch.float64))
    return jagged_from_list(out, offsets)

@torch.compile(backend="aot_eager")
def f(nt):
    nested_size = ((2, 3, 4), 5)
    offsets = None
    nt2, _ = get_jagged_tensor(nested_size, offsets)
    nt3 = torch.cat([nt2, nt], dim=-1)
    return nt3.sin() * nt3.size(1)

nested_size = ((2, 3, 4), 5)
offsets = None
nt, _ = get_jagged_tensor(nested_size, offsets)
out = f(nt)