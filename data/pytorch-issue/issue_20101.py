import torch.nn as nn

from torch import nn
import torch.jit

model = nn.Sequential(nn.Conv2d(2, 2, 1, 1, 1))

torch.jit.trace(model.forward, torch.randn(1, 1, 2, 2))

model = nn.Sequential(nn.Conv2d(1, 1, 3))
torch.jit.trace(model, torch.randn(1, 1, 3, 3))