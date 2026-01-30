import torch
import torch.nn as nn
x = torch.randn(1024, 1024).cuda()
for _ in range(100000):
    y = torch.matmul(x, x)
print('workload test done')