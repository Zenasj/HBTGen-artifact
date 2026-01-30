import torch
import torch.nn as nn

from torch import nn
from torch.nn.utils.weight_norm import weight_norm

x = nn.Conv2d(2, 2, 3)
weight_norm(x)
x.weight_g.data.zero_()
print(x.weight_g)
print(x.weight_v)
print(x.weight)