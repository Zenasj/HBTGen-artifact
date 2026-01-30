import torch.nn.functional as F

import torch
from torch.nn import fucntional as F

y = torch.randn(2, 3)
t = torch.tensor([-1, 2])

torch.vmap(F.cross_entropy)(y, t, ignore_index=-1)