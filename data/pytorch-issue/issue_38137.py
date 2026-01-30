import torch

for x in dir(torch._C):
    c = getattr(torch._C, x)
    if type(c) is type(torch):
        sys.modules[f'torch._C.{x}'] = c