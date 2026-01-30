import torch.nn as nn

import torch
import torch.nn.functional as F

#device = "cpu"
device = "mps"
projected = torch.rand([8]).to(device)
x = torch.rand([1, 3598, 4, 8]).to(device)
linear = F.linear(x, projected)