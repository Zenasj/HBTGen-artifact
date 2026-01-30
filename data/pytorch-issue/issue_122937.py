import torch
import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True

@torch.compile()
def f(x, il, jl):
    i0, i1, i2 = il.tolist()
    j0, j1, j2 = jl.tolist()
    for i in [i0, i1, i2, j0, j1, j2]:
        torch._check_is_size(i)
    r = torch.zeros(x.item())
    return torch.ops.aten.split_with_sizes.default(r, [i0 + j0, i1 + j1, i2 + j2])

f(torch.tensor(20), torch.tensor([2, 5, 3]), torch.tensor([3, 4, 3]))