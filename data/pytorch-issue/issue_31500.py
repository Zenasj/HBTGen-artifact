import torch.nn as nn

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
conv = torch.nn.ConvTranspose2d(1, 1, 1, 1, bias=False).cuda().half()
input_large = torch.randn(1024, 1, 1024, 1024, dtype=torch.half, device='cuda')
ret = conv(input_large)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True