py
import torch
import torch._refs
from torch._prims.executor import make_traced

def foo(a):
    return torch._refs.mean(a, keepdim=False)

a = torch.randn(3, 3, device='cuda')
make_traced(foo)(a, executor="nvfuser")