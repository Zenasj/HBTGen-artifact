import torch
import pickle
from torch.utils import benchmark
from itertools import product

device = 'cpu'
dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
n = (100, 200, 500, 1000, 10000)

result = []

for dtype, num in product(dtypes, n):
    x = torch.rand(num, dtype=dtype, device='cpu')
    torch.digamma(x)
    stmt = 'torch.digamma(x)'
    measurement = benchmark.Timer(
        stmt=stmt,
        globals={'x': x},
        label=stmt + " Benchmark",
        sub_label=f"dtype {dtype} - n : {num}",
        description="vectorized",
    ).blocked_autorange()

    result.append(measurement)

fname_prefix = "benchmark_digamma_"

benchmark.Compare(result).print()
with open(fname_prefix+"vectorized", "wb") as f:
    pickle.dump(result, f)