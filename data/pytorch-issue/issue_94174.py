import torch
import numpy as np


def fn(input):
    max_ratio = np.abs(np.log(4))
    dwh = input.clamp(min=-max_ratio, max=max_ratio)
    return dwh

x = torch.rand([1]).cuda()

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn, backend='eager')
ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')