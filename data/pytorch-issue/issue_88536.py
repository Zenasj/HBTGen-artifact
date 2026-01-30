import torch.nn as nn

import torch
import torch.nn.functional as F

device = torch.device('cuda')
t = torch.ones(2, 3, 240, 320, device=device, dtype=torch.bfloat16)
F.interpolate(t, scale_factor=0.5, mode='nearest')
F.interpolate(t, scale_factor=0.5, mode='bilinear')