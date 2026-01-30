import torch

from timeit import default_timer
from itertools import product
from torch.utils.benchmark import Measurement, Timer

def bench_mask_fill(
    n,
    binary_func,
    dtype=torch.float32,
) -> Measurement:
    t = Timer(
        stmt=f"x.masked_fill(y, -17.0); torch.mps.synchronize()",
        setup=f"x,y = torch.rand(1, 20, {n}, {n}, dtype={dtype}, device='mps'), torch.ones({n}, {n}, device='mps').triu().bool()",
        globals = {'f': binary_func},
        language="python", timer=default_timer
    )
    return t.blocked_autorange()

if __name__ == "__main__":
    n = 1024
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        eager_t = bench_mask_fill(n, torch.fmax, dtype)
        use_msec = eager_t.mean > 1e-4
        multiplier = 1e3 if use_msec else 1e6
        uname = "msec" if use_msec else "usec"
        print(f"torch.masked_fill_() {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname}")

import torch

from timeit import default_timer
from itertools import product
from torch.utils.benchmark import Measurement, Timer

def bench_mask_fill(
    n,
    binary_func,
    dtype=torch.float32,
) -> Measurement:
    t = Timer(
        stmt=f"x.masked_fill(y, -17.0); torch.mps.synchronize()",
        setup=f"x,y = torch.rand(1, 20, {n}, {n}, dtype={dtype}, device='mps'), torch.ones({n}, {n}, device='mps').triu().bool()",
        globals = {'f': binary_func},
        language="python", timer=default_timer
    )
    return t.blocked_autorange()

if __name__ == "__main__":
    n = 1024
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        eager_t = bench_mask_fill(n, torch.fmax, dtype)
        use_msec = eager_t.mean > 1e-4
        multiplier = 1e3 if use_msec else 1e6
        uname = "msec" if use_msec else "usec"
        print(f"torch.masked_fill_() {str(dtype):>14} {eager_t.mean*multiplier:>7.2f} {uname}")