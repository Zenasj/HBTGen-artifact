import torch.nn as nn

import torch

x = torch.randn(1, 10, 10, 10, device="mps")
c = torch.nn.Conv3d(1, 1, 3, device="mps")
c(x)