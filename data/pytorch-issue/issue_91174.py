py
import torch
from functorch import vmap
x = torch.randn(32, 2, 3)
y = vmap(torch.nansum, (0, None))(x, 1)
#  UserWarning: There is a performance drop because we have not yet implemented the batching rule