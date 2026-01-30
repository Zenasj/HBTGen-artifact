import torch.nn as nn

import torch

self = torch.full((0, 0, 9, 15, 0,), 0, dtype=torch.int64, requires_grad=False)
target = torch.full((0, 0, 9, 15, 0,), 0.5, dtype=torch.float64, requires_grad=False)
weight = torch.full((0,), 5.36871e+08, dtype=torch.float64, requires_grad=False)
reduction = 1
ignore_index = -1250999896764
label_smoothing = 0
torch._C._nn.cross_entropy_loss(self, target, weight, reduction, ignore_index, label_smoothing)