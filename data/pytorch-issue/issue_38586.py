import torch
from torch.autograd import gradcheck

mat = torch.randn(2, 2, dtype=torch.float64)
vec = torch.randn(1, dtype=torch.float64).expand(2).requires_grad_(True)

def fn(vec):
    return (mat @ vec).sum()

gradcheck(fn, (vec,))