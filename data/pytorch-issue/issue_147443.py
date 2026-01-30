import torch.nn as nn

import torch
import torch.nn.functional as F

device = torch.device('mps')

## To provoke the error, an non-continguous tensor needs to be created
q = torch.rand(3, 592, 4, 49, 32).to(device)
k = torch.rand(3, 592, 4, 49, 32).to(device)
v = torch.rand(3, 592, 4, 49, 32).to(device)

x = F.scaled_dot_product_attention(q, k, v)