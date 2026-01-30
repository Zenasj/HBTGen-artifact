import numpy as np

import torch

@torch.compile
def f(x):
    x_view = x.view(dtype=torch.int16)
    return x_view.mul(2)

x = torch.ones(4, dtype=torch.float16, device='cuda')
out = f(x)

def f(fp16_inp):
    out = fp16_inp.sum()  # do something with the fp16 that **does** require upcasting
    return fp16_inp.view(dtype=torch.int16) + out.view(dtype=torch.int16)