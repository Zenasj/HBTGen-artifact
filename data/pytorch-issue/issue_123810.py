import torch

device = 'cuda'

def fn(x):
    _, L = x.shape
    return torch.full((L, L), torch.finfo(torch.float16).min, device=device)

cfn = torch.compile(fn, dynamic=True)

import functools
input_fn = functools.partial(torch.randint, 10, 1000, device=device)

cfn(input_fn((2, 3)))
cfn(input_fn((2, 4)))  # expect don't recompile here
cfn(input_fn((2, 5)))  # expect don't recompile here