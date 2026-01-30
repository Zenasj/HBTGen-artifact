import torch.nn as nn

py
import torch
import torch.nn.functional as F
from time import time

start = time()

for x in range(1):
    result = F.conv_transpose1d(torch.randn(1, 1026, 224), torch.randn(1026, 1, 1024), stride=256, padding=0)

print((time()-start)/1)

import torch
import torch.nn.functional as F
from time import time

def f(x):
    return F.conv_transpose1d(x, torch.randn(1026, 1, 1024), stride=256, padding=0)


f(torch.randn(1, 1026, 224))

n = 10
start = time()
for i in range(n):
    f(torch.randn(1, 1026, 500*(i+1)))

print((time()-start)/n)