import torch

y = torch.logspace(0, 3, 4, device='cuda', dtype=torch.float32).to(torch.int)
# output: tensor([  1,  10,  99, 999], device='cuda:0', dtype=torch.int32)