import torch.nn as nn

import torch
import torch.nn.functional as F
x = torch.randn(3, 3)
y = torch.randn(9)
F.logsigmoid(x, out=y) # crashes