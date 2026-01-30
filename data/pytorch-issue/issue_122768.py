import torch
import triton

@triton.jit
def test_kernel(kernel):
    pass

@torch.compile(backend="eager")
def f(x):
    grid = (x.numel(),)
    # test_kernel[grid](kernel=x)
    with torch.cuda.device(x.device.index):
        test_kernel[grid](kernel=x)

t1 = torch.rand(5, device="cuda")
f(t1)