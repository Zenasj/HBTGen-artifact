import torch

a = torch.rand(100,4).cuda()
torch.use_deterministic_algorithms(True, warn_only=True)
a[[64]] = 1.0