py
import torch
from functorch import vmap
x = torch.randn(4, 3, 2)
z = vmap(torch.linalg.lu_factor)(x)
#  UserWarning: There is a performance drop because we have not yet implemented the batching rule