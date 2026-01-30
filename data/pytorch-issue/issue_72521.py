import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.randn(1, 2, 3)
F.pad(x, [0, 0, 0, 0, 1, 1], mode='reflect')