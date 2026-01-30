import torch.nn as nn

import torch
from torch.utils.checkpoint import checkpoint

with torch.device('meta'):
    m = torch.nn.Linear(20, 30)
    x = torch.randn(1, 20)

out = checkpoint(m, x, use_reentrant=False, preserve_rng_state=False)