py
import torch
from functorch import vmap
x = torch.randn(32, 2, 3)
y = torch.randn(32, 2, 3)
z = vmap(torch.complex)(x, y)
#  UserWarning: There is a performance drop because we have not yet implemented the batching rule