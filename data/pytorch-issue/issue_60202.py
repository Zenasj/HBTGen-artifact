import torch
import time

# 64 MB tensors
x = torch.rand(1, 256, 256, 256)
y = torch.rand(1, 256, 256, 256)

NITER = 10000

s = time.time()
for i in range(NITER):
    x + y
elapsed_float = time.time() - s
ms_per_iter_float = elapsed_float / NITER * 1000

print(ms_per_iter_float)

import torch
import time

x = torch.rand(1, 256, 256, 256)
y = torch.rand(1, 256, 256, 256)
z = torch.rand(1, 256, 256, 256)

NITER = 10000

s = time.time()
for i in range(NITER):
    torch.add(x, y, out=z)
elapsed_float = time.time() - s
ms_per_iter_float = elapsed_float / NITER * 1000

print(ms_per_iter_float)