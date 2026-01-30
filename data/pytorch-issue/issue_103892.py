import torch.nn as nn

import torch
import torch.nn.functional as F

def func(x):
    return F.adaptive_max_pool1d(x, 1)

opt_func = torch.compile(func, dynamic=True)

x = torch.randn((1, 4), device='cuda')
opt_func(x)

x = torch.randn((1, 5), device='cuda')
opt_func(x)