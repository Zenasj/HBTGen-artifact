import torch.nn as nn

import torch
from torch.nn.utils.parametrizations import weight_norm

c1 = torch.nn.Conv1d(100, 10, 10)
c1 = weight_norm(c1, dim=0)

c1 = torch.compile(c1, fullgraph=True)
x = torch.zeros(1, 100, 20)
c1(x)