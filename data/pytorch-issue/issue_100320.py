import torch
from torch import vmap

@torch.compile
def norm(g):
    return torch.linalg.norm(g)

norm(torch.ones(3)) # => works, tensor(1.7321)

# fails with error in log
xs = torch.ones(10, 3)
vmap_norm = vmap(norm)
vmap_norm(xs)