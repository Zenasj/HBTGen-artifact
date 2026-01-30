import torch

@torch.compile(backend="eager", dynamic=True)
def f(t, new_size, new_stride):
    tmp = t.as_strided(new_size, new_stride)
    tmp = tmp.view(-1)
    return t * tmp.sum()

x = torch.randn(3)
new_size = [0, 3]
new_stride = [3, 1]
out = f(x, new_size, new_stride)