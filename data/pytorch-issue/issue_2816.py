import torch
import time

N, C, H, W = 64, 256, 64, 64
repetitions = 100

x = torch.randn(N, C, H, W).cuda()
w = torch.randn(C).cuda()

t0 = time.time()
for _ in range(repetitions):
    y = x * w.view(1, C, 1, 1)
    torch.cuda.synchronize()
print (time.time() - t0) / repetitions

w = torch.randn(N).cuda()
...
y = x * w.view(N, 1, 1, 1)

w = torch.randn(W).cuda()
...
y = x * w.view(1, 1, 1, W)

w = torch.randn(H).cuda()
...
y = x * w.view(1, 1, H, 1)

import torch
import time

N, C, H, W = 64, 256, 64, 64
repetitions = 100

x = torch.randn(N, C, H, W).cuda()
w = torch.randn(C).cuda().expand(N, C).contiguous()

t0 = time.time()
for _ in range(repetitions):
    y = x * w.view(N, C, 1, 1)
    torch.cuda.synchronize()
print( (time.time() - t0) / repetitions )