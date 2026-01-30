import torch
from torch.testing._internal.triton_utils import add_kernel


@torch.compile
def f(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    n_elements = output.numel()
    grid = (x.numel(),)
    add_kernel.run(x, y, output, n_elements, warmup=False, grid=grid, BLOCK_SIZE=16)
    return output

with torch.inference_mode():
    t1 = torch.rand(5, device="cuda")
    t2 = torch.rand(5, device="cuda")
    o1 = torch.zeros_like(t1)
    outs = f(t1, t2, o1)