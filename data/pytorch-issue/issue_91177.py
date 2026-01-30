py
import torch
from functorch import vmap
x = torch.randn(4, 3, 3)
index = torch.tensor([0, 2])
z = vmap(torch.index_fill, (0, None, None, None))(x, 1, index, -1)
#  UserWarning: There is a performance drop because we have not yet implemented the batching rule