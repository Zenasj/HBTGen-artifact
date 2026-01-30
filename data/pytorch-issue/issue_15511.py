import torch.nn as nn

import math
import torch #version 1.0
import torch.nn.functional as F

sizes = [100, 1000, 3000, 5000, 7500, 8500, 9999, 10000]
for size in sizes:
    print("[{}] Start".format(size))
    a = torch.randn(size, 20).float().cuda()
    print("Before")
    # The operation
    b = F.pdist(a, p=2)
    print("After")
    a = torch.randn(size, 20).float().cuda()
    print("---End---")

import torch
n = torch.tensor(2 ** 15, dtype=torch.int64)
truth = torch.cat([torch.full((n - i - 1,), i, dtype=torch.int64) for i in range(n)])
n2 = n.to(torch.float64) - .5
k = torch.arange(n * (n - 1) // 2)
test = (n2 - torch.sqrt(n2 * n2 - 2 * k.to(torch.float64) - 1)).to(torch.int64)
assert torch.all(truth == test)