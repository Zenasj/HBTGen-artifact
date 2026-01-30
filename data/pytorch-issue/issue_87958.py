import torch.nn as nn

import torch
import torch.nn.functional as F
input = torch.randn(1, 3, 5, 5)
weight = torch.randn(1, 3, 1, 1)
output = F.prelu(input, weight)
print(output)