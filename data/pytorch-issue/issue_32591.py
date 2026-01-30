3
a = torch.rand(1, 1, 16, 2, 16, 2, 16, 2, 2, 2, 2, device = "cuda")
b = torch.rand(729, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, device = "cuda")

# Warmup
for i in range(100): output1 = (a * b).sum(dim = (-3, -2, -1))

# Method 1
with torch.autograd.profiler.profile(use_cuda = True) as prof:
  output1 = (a * b).sum(dim = (-3, -2, -1))
print(prof.key_averages().table(sort_by="cuda_time_total"))

# Warmup
for i in range(100):
  (a2, b2) = torch.broadcast_tensors(a, b)
  output2 = torch.einsum("...ijk, ...ijk -> ...", a2, b2)

# Method 2
with torch.autograd.profiler.profile(use_cuda = True) as prof:
  (a2, b2) = torch.broadcast_tensors(a, b)
  output2 = torch.einsum("...ijk, ...ijk -> ...", a2, b2)
print(prof.key_averages().table(sort_by="cuda_time_total"))

import torch
from torch.utils import benchmark

results = []
for b in [10, 10000, 2000000]:
    for n in [10, 100, 10000, 1000000]:
        if b * n >= 1000000000:
            continue

        description = f'[{b}, {n}]'

        x = torch.rand(b, n, device='cuda')
        y = torch.rand(b, n, device='cuda')

        results.append(benchmark.Timer(
            stmt='(x * y).sum(dim=-1)',
            globals={'x': x, 'y': y},
            description=description,
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch.einsum("...j,...j->...", x, y)',
            globals={'x': x, 'y': y},
            description=description,
        ).blocked_autorange())

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()


results = []
for b in [10, 100, 1000]:
    for n in [10, 100, 10000, 1000000]:
        if b * b * n >= 1000000000:
            continue

        description = f'[{b}, {b}, {n}]'

        x = torch.rand(b, b, n, device='cuda')
        y = torch.rand(b, b, n, device='cuda')

        results.append(benchmark.Timer(
            stmt='(x * y).sum(dim=-1)',
            globals={'x': x, 'y': y},
            description=description,
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch.einsum("...j,...j->...", x, y)',
            globals={'x': x, 'y': y},
            description=description,
        ).blocked_autorange())

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()


results = []
for b in [10, 100, 1000]:
    for n in [10, 100, 10000, 1000000]:
        if b * b * n >= 1000000000:
            continue

        description = f'[{b}, {b}, {n}]'

        x = torch.rand(b, 1, n, device='cuda')
        y = torch.rand(1, b, n, device='cuda')

        results.append(benchmark.Timer(
            stmt='(x * y).sum(dim=-1)',
            globals={'x': x, 'y': y},
            description=description,
        ).blocked_autorange())

        results.append(benchmark.Timer(
            stmt='torch.einsum("...j,...j->...", x, y)',
            globals={'x': x, 'y': y},
            description=description,
        ).blocked_autorange())

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

import numpy as np

import torch
import torch.utils.benchmark as benchmark
from torch.testing._internal.common_utils import make_tensor


def generate_test_cases_from_issue_32591(device, dtype):
    A = make_tensor((1, 1, 16, 2, 16, 2, 16, 2, 2, 2, 2), device, dtype)
    B = make_tensor((729, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2), device, dtype)

    yield '...ijk,...ijk->...', A, B


results = []

for equation, *operands in generate_test_cases_from_issue_32591('cpu', torch.float):
    print('equation:', equation, '\toperands:', [op.shape for op in operands])

    sub_label = equation

    results.append(benchmark.Timer(
        stmt='(operands[0] * operands[1]).sum(dim=(-3, -2, -1))',
        globals={'equation': equation, 'operands': operands},
        sub_label=sub_label,
        description='mul/sum',
    ).blocked_autorange(min_run_time=1))

    results.append(benchmark.Timer(
        stmt='torch.einsum(equation, *operands)',
        globals={'equation': equation, 'operands': operands},
        sub_label=sub_label,
        description='torch.einsum',
    ).blocked_autorange(min_run_time=1))

    results.append(benchmark.Timer(
        stmt='numpy.einsum(equation, *operands)',
        setup='import numpy',
        globals={'equation': equation, 'operands': [op.cpu().numpy() for op in operands]},
        sub_label=sub_label,
        description='numpy.einsum',
    ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize(rowwise=True)
compare.print()