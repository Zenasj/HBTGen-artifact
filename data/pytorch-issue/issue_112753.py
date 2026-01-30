import torch

def func(x):
    batch_shape = x.shape[:1]
    out = torch.cat([x.new_zeros(1).expand(batch_shape + (1,)), x], dim=-1)
    return out


cfunc = torch.compile(func)

x = torch.randint(0, 256, size=(3, 255), dtype=torch.uint8)
out = cfunc(x)