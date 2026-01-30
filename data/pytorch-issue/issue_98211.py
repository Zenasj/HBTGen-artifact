import torch.nn as nn

import torch

x = torch.randn(2, 32, device="mps")
l = torch.nn.Linear(64, 4, device="mps")
y = l(x)