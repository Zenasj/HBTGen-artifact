import torch.nn as nn

import torch

def fn(input):
    return torch.nn.functional.normalize(input, dim=0, out=None)

x = torch.rand([1])

fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
compiled = compiled(x)
print('==== torchcomp mode OK! ====')