import torch.nn as nn
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as config
config.coordinate_descent_tuning = True

def bench(f, name=None, iters=1000, warmup=5, display=True, profile=False):
    import time
    from triton.testing import do_bench

    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    us_per_iter = do_bench(lambda: f())*1000

    if name is None:
        res = us_per_iter
    else:
        res= f"{name}: {us_per_iter:.3f}us"

    if display:
        print(res)
    return res

def count_bandwidth(f, inp):
    out = f()
    mem_bytes = inp.numel() * inp.dtype.itemsize + out.numel() * out.dtype.itemsize
    us = bench(f, display=False)
    print((1e6/us) * mem_bytes / 1e9)

with torch.no_grad():
    for D in range(8192, 8192*11+1, 8192): #256
        inp = torch.randn(2048*32, D, dtype=torch.bfloat16, device='cuda')
        mod = nn.LayerNorm([D], dtype=torch.bfloat16, device='cuda')

        print(D)
        count_bandwidth(lambda: mod(inp), inp)

        torch._dynamo.reset()
        comp_mod = torch.compile(mod, dynamic=False)
        count_bandwidth(lambda: comp_mod(inp), inp)

        torch._dynamo.reset()
        comp_mod = torch.compile(mod, dynamic=False)
        count_bandwidth(lambda: comp_mod(inp), inp)

config._temp_welford_reduction_div_approx = False