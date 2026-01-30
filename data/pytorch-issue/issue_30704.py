import torch
x = torch.randn(3,3)
x.norm(dim=(0,1)), x.cuda().norm(dim=(0,1))