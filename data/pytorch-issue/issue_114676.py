import torch

@torch.compile(fullgraph=True, dynamic=True, backend="inductor")
def test(x, mul, dim=-1):
    size = x.size(dim)
    m = size/mul
    if m.is_integer():
        return m
    return size

t = torch.randn((3, 6, 4, 2), requires_grad = False)
test(t, 2)