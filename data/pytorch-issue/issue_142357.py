import torch
torch.set_default_device('cuda')
from triton.testing import do_bench


class DoNothing:
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

@torch.compile
def f(x):
    with DoNothing():
        x = x* 2
    return x

f(torch.randn(5))