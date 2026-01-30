import torch
from torch._refs import cat
import torch._prims as prims

t = torch.ones(4, 4, 4, 4)
s = torch.ones(4,)
cat([t, s])  # Errors

t = prims.utils.TensorMeta(t)
s = prims.utils.TensorMeta(s)
cat([t, s])  # does not error