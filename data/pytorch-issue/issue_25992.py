import torch

@torch.jit.script
def f(a:int, b:int):
    return max([a, b])

In [13]: torch.__version__
Out[13]: '1.3.0.dev20190911'