py
import torch
import torch._refs
from torch._prims.executor import make_traced
a = torch.ones(3, 3, dtype=torch.bool, device='cuda')
b = torch.randn(3, 3, device='cuda')
func = lambda a, b: torch._refs.where(a, 0.0, b)
assert make_traced(func)(a, b, executor="nvfuser").dtype == torch.float32