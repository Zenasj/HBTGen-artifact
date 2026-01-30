import torch
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare
from itertools import product

results = []
num_threads = 1

size = 5000

beta = 1.2
alpha = 2.2

dtypes = torch.testing.get_all_fp_dtypes()
devices = [torch.device('cpu'), torch.device('cuda')]

for device, dtype in product(devices, dtypes):
    if torch.__version__ == '1.6.0' and device == torch.device('cuda') and dtype == torch.bfloat16:
        continue
    if torch.__version__ == '1.6.0' and device == torch.device('cpu') and dtype == torch.half:
        continue

    M = torch.randn(size, size).to(device=device, dtype=dtype)
    vec1 = torch.randn(size).to(device=device, dtype=dtype)
    vec2 = torch.randn(size).to(device=device, dtype=dtype)

    tasks = [("torch.addr(M, vec1, vec2, beta=beta, alpha=alpha)", f"{dtype}")]
    timers = [Timer(stmt=stmt,
                    num_threads=num_threads,
                    label=f"{torch.__version__}, size: {size}",
                    sub_label=f"{device}",
                    description=dtype,
                    globals=globals()) for stmt, dtype in tasks]

    repeats = 3
    for i, timer in enumerate(timers * repeats):
        results.append(timer.blocked_autorange())
        print(f"\r{device}-{dtype} {i + 1} / {len(timers) * repeats}")

comparison = Compare(results)
comparison.print()