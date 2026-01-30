py
import torch
from torch._vmap_internals import vmap
import warnings
warnings.simplefilter('ignore')

device = 'cuda:0'
x = torch.randn(4, requires_grad=True, device=device)
indices = torch.tensor([0, 1], device=device)
y = x.gather(0, indices)

def vjp(v):
    return torch.autograd.grad(y, x, v, retain_graph=True)

vs = torch.randn(2, *y.shape, device=device)
vmap(vjp)(vs)
vmap(vjp)(vs)
vmap(vjp)(vs)

# This script has the following output, despite the warnings filter
# [W BatchedFallback.cpp:63] Warning: Batching rule not implemented for aten::gather_backward falling back to slow (for loop and stack) implementation (function batchedTensorForLoopFallback)
# [W BatchedFallback.cpp:63] Warning: Batching rule not implemented for aten::gather_backward falling back to slow (for loop and stack) implementation (function batchedTensorForLoopFallback)
# [W BatchedFallback.cpp:63] Warning: Batching rule not implemented for aten::gather_backward falling back to slow (for loop and stack) implementation (function batchedTensorForLoopFallback)