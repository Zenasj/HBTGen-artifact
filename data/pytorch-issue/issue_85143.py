import torch.nn as nn

import torch
import torch.nn.functional as F

dev_mps = torch.device('mps')
dev_cpu = torch.device('cpu')

dev=dev_cpu
a = torch.rand(1, 2, 3).to(dev)
out = F.pad(a, [0, 0, 0, 1], "constant", value=0.0)

print (out.shape)

dev=dev_mps
a = torch.rand(1, 2, 3).to(dev)
out = F.pad(a, [0, 0, 0, 1], "constant", value=0.0)
print (out.shape)