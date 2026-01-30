import torch
from torch import vmap

def fn(a):
    return torch.ops.aten._is_all_true(a)

a = torch.tensor([[True], [False]])

result = vmap(fn)(a)