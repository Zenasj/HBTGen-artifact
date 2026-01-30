import torchvision

shift_size = shift_size.copy()

import torch
from torchvision.models import swin_v2_s

mod = swin_v2_s().cuda()
opt_mod = torch.compile(mod, fullgraph=True, dynamic=False)

inp = torch.randn(4, 3, 224, 224).cuda()
opt_mod(inp)