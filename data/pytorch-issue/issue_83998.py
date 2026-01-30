import torch.nn as nn

import torch
from torch import nn

with torch.no_grad():
  a = nn.Parameter(torch.rand(10))
print(a.requires_grad())