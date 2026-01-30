import torch

@torch.compile(dynamic=True)
def toy_example(x):
    y = x.sin()
    z = y.cos()
    return y, z

toy_example(torch.randn([8192, 1024], device="cuda"))