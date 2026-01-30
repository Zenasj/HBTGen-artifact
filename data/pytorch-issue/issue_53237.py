import torch

t = torch.randn(5, 5, requires_grad=True, dtype=torch.double)
# idx = torch.tensor([3, 0]) # Works.
idx = torch.tensor([2, 2])

print(torch.unique(idx))

v = torch.tensor(-2.5122, requires_grad=True, dtype=torch.double)

def fn(v):
    return t.index_fill(0, idx, v)

torch.autograd.gradcheck(fn, (v,))
torch.autograd.gradgradcheck(fn, (v,))