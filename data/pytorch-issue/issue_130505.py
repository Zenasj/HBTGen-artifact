import torch

@torch.compile(fullgraph=True)
def f(values, offsets):
    nt = torch.nested.nested_tensor_from_jagged(values, offsets)
    return nt + 1

# called like: g(nt, torch.randn_like(nt.values()), nt.offsets())
@torch.compile(fullgraph=True)
def g(nt1, values2, offsets):
    nt2 = torch.nested.nested_tensor_from_jagged(values2, offsets)
    return nt1 + nt2