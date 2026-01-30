import torch
import torch.nn as nn

from torch import nn
from torch.nn.utils import prune


m = nn.Linear(4, 2)
prune.l1_unstructured(m, 'weight', amount=0.2)
param_names = []

for n, p in m.named_parameters():
    param_names.append(n)

print(param_names)

[weight, bias]

[bias, weight_orig]