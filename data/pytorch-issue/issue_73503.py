import torch.nn as nn

import torch
input = torch.rand([0, 1])
torch.nn.init.orthogonal_(input)
# ZeroDivisionError: integer division or modulo by zero