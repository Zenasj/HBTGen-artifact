import torch
from triton.testing import do_bench

x = torch.randn(2**30, device='cuda')

ms = do_bench(lambda: x.sum(dim=-1))

bandwidth_gbyte = x.numel() * x.dtype.itemsize / (10**9)

time_s = ms / 1000

bw_per_second = bandwidth_gbyte / time_s

print(bw_per_second)

import torch
from triton.testing import do_bench

x = torch.randn(2**9, 2**20, device='cuda')

ms = do_bench(lambda: x.sum(dim=-1))

bandwidth_gbyte = x.numel() * x.dtype.itemsize / (10**9)

time_s = ms / 1000

bw_per_second = bandwidth_gbyte / time_s

print(bw_per_second)