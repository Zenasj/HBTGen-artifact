import torch
import time
import torch.nn as nn

d = 3
m = nn.Conv2d(256, 256, 3, 1, padding=d, dilation=d, bias=False)
m.cuda()
total = 0.0

for _ in range(100):
    i = torch.rand(1, 256, 80, 45).cuda()
    s = time.time()
    r = m(i)
    torch.cuda.synchronize()
    e = time.time()
    total += (e-s)
print(torch.version.__version__)
print(total/100)

import time
import random
import torch
import torch.nn as nn
# uncomment this to gain performance:
# torch.backends.cudnn.deterministic=True

dilation = 4
conv = nn.Conv2d(47, 64, 3, padding=dilation, dilation=dilation, stride=1).cuda()
def mk_input():
  return torch.rand((random.randint(2, 15), 47, 24, 40), device='cuda:0')

for x in range(10):
  outp = conv(mk_input())

a = time.perf_counter()
for x in range(1000):
  outp = conv(mk_input())
b = time.perf_counter()
print(b-a)
# gives 0.475 with deterministic=True, and 
# 1.615 with deterministic=False