import torch
B, P, R, M = 1, 10, 10, 5
x1 = torch.randn(B, P, M, dtype=torch.float32)
x2 = torch.randn(B, R, M, R, M, dtype=torch.float32)
dist = torch.cdist(x1, x2)
print(x1.shape, x2.shape, dist.shape)

import torch
B, P, R, M = 1, 10, 10, 5
x1 = torch.randn(B, P, M, R, M, dtype=torch.float32)
x2 = torch.randn(B, R, M, R, M, dtype=torch.float32)
dist = torch.cdist(x1, x2)
print(x1.shape, x2.shape, dist.shape)