import random

import time
import torch
import numpy as np

data = [np.random.rand(8, 800, 1333) > 0.5 for _ in range(2)]
dtype_np = np.bool
dtype_pt = torch.bool

def f1():
    return np.asarray(data, dtype=dtype_np)

def f2():
    return np.stack(data)

def f3():
    return torch.as_tensor(data, dtype=dtype_pt)

def f4():
    return torch.stack([torch.from_numpy(x) for x in data])

def benchmark(f, iter, warmup):
    for k in range(warmup): f()
    start = time.perf_counter()
    for k in range(iter): f()
    torch.cuda.synchronize()
    return time.perf_counter() - start

print(benchmark(f1, 10, 1))
print(benchmark(f2, 10, 1))
print(benchmark(f3, 10, 1))
print(benchmark(f4, 10, 1))

0.2839105408638716
0.013459203764796257
20.43221192806959
0.012655530124902725