3
import functorch
import torch

def cat(x, y):
    return torch.cat((x,y), dim=0)

def concatenate(x,y):
    return torch.concatenate((x,y), dim=0)

v = torch.rand(2,16)

# Works
print(functorch.vmap(cat)(v, v))

# RuntimeError
print(functorch.vmap(concatenate)(v, v))

import torch
import functorch

def cat(x, y):
    return torch.cat((x,y), dim=0)

def concatenate(x,y):
    return torch.concatenate((x,y), dim=0)

v = torch.rand(2,16)

print(functorch.vmap(cat)(v, v).shape)
# Output: torch.Size([2, 32])

print(functorch.vmap(concatenate)(v, v).shape)
# Output: torch.Size([2, 32])

print( torch.__version__)
# Output:  '2.0.0a0+git859ac58'