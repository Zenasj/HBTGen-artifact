import torch.nn as nn

import torch
from torch.autograd import gradcheck

torch.set_default_dtype(torch.double)

device='cuda'

x = torch.randn(2, 7, 7, requires_grad=True, device=device)
samples = x.new(1, 2, 2).uniform_()

def fn(x):
    return torch.nn.functional.fractional_max_pool2d(
                x, (2, 2), output_size=(3, 3), _random_samples=samples)

gradcheck(fn, [x])