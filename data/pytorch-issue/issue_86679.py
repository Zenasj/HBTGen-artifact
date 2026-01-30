import torch.nn as nn

import torch
import torch.nn.functional as F

image = torch.randn(1, 4, 32, 32).to(device="cuda", dtype=torch.bfloat16)
out = F.interpolate(image, size=(64, 64), mode="nearest")