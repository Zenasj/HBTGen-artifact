import torch
from torch._inductor.utils import do_bench_using_profiling
from triton.testing import do_bench

torch._inductor.config.force_disable_caches = True
# torch._inductor.config.aggressive_fusion = True

import os


def test(x):
    y = torch.sum(x)
    z = x / 10.0
    z_t = z.t().contiguous().t()
    return y, z, z_t

test = torch.compile(test)

x = torch.randn(4096, 2048, device="cuda")

y, z, z_t = test(x)

print(y.shape, z.shape, z_t.shape)