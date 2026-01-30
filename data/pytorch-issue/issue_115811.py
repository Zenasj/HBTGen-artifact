import torch.nn as nn

import torch

def err(x):
        return torch.rrelu(x)
#       return torch.nn.functional.rrelu(x) # does not work either
#       return torch.nn.functional.rrelu(x, 1/8, 1/3, False, False) # does not work either

inp = torch.rand(1, 2, 3)

print('Eager', err(inp))
print('Compiled', torch.compile(err)(inp))