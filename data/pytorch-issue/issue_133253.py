import torch

from triton.runtime.jit import reinterpret as tl_reinterpret
import triton.language as tl

@torch.compile(backend="eager", fullgraph=True)
def fn(x):
    return tl_reinterpret(x, dtype=tl.float8e4nv)


fn(torch.randn(3, device='cuda'))

def fn(x):
    return tl_reinterpret(x, dtype=tl.float8e4nv)


x = torch.randn(3, device='cuda')
fn(x)