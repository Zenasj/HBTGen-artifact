import torch.nn as nn

import torch
x = torch.rand(16, 8, 24).bfloat16().cuda()
torch.nn.Upsample(80, mode="linear", align_corners=True)(x)