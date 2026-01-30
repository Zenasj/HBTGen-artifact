import torch.nn as nn

python
import argparse
import copy
import torch
from torch import nn
from torch.utils._benchmark import Timer


parser = argparse.ArgumentParser()
parser.add_argument('--sync', action='store_true')
args = parser.parse_args()

# Initialization
a = torch.randn(1024, 2, 256, device='cuda')
b = torch.randn(1024, 2, 256, device='cuda')

self_attention = nn.MultiheadAttention(256, 8).to('cuda')
layers = nn.ModuleList([copy.deepcopy(self_attention) for _ in range(10)])


def run_test(a, b, layers):
    for layer in layers:
        if args.sync:
            torch.cuda.synchronize()
        a = layer(a+b, a+b, a)[0]


# Timing from torch.utils._benchmark.Timer
timer = Timer(stmt='run_test(a, b, layers)', globals=globals())
t = timer.timeit(number=3)._median*1e3

# Profiling with torch.autograd.profiler.profile(use_cuda=True)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    run_test(a, b, layers)

print(prof.table(sort_by='cpu_time', row_limit=20))
print(f"Recorded timeit time: {t: .4f} ms")