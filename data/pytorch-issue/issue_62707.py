import torch
import einops

def f(x):
    return x.shape[0] // 4
test_tensor = torch.rand([1, 2, 376, 16, 16])
res = torch.jit.trace(f, test_tensor)

@torch.jit.script
def f(x):
    return torch.div(torch.tensor([x.shape[0]]), 4, rounding_mode='floor')