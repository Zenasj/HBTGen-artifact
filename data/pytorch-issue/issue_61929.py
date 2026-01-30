import torch

A = torch.randn(2, 3, 4096).cuda()
B = torch.randn(2, 3, 3).cuda()
X = torch.linalg.solve(B,A)

torch.cuda.synchronize()