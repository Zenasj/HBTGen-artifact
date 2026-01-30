import time
import torch

data = torch.rand(2000000, dtype=torch.float32, device=torch.device('cuda:0'))
num_topk = 1000

def topk1():
    return data.topk(num_topk, sorted=False)

def topk2():
    sort, idx = data.sort(descending=True)
    return sort[:num_topk], idx[:num_topk]

def benchmark(f, iter, warmup):
    for k in range(warmup): f()
    start = time.perf_counter()
    for k in range(iter): f()
    torch.cuda.synchronize()
    return time.perf_counter() - start

print(benchmark(topk1, 100, 3))
print(benchmark(topk2, 100, 3))
print(benchmark(topk1, 100, 3))
print(benchmark(topk2, 100, 3))

import torch
t = torch.randn(1, 3, 224, 224, device="cuda:0")