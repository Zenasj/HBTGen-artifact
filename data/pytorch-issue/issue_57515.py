a[mask] = val

a[mask] = tensor

import torch
from torch.utils.benchmark import Timer, Compare

results = []
for device in ['cuda', 'cpu']:
    for dtype in [torch.float32]:
        for tensor_size in [100_000, 500000, 1_000_000, 5000000, 10_000_000]:
            if tensor_size >5000000 and device=='cpu':
                continue
            for mask_true_ratio in [0, 0.25, 0.5, 0.75, 1.]:
                a = torch.rand(tensor_size, dtype=dtype, device=device)
                mask = torch.empty(tensor_size, dtype=torch.bool, device=device).bernoulli_(mask_true_ratio)
                size = (1000, tensor_size//1000)
                val=1.
                stmts = ("a[mask]=val", "a.masked_fill_(mask, val)")
                labels = (f"index, {device}", f"masked_fill, {device}")
                timers = [Timer(stmt=stmt, num_threads = 10, sub_label=f"{tensor_size}, full {mask_true_ratio}", label = "Fill", description = l,
                globals = globals()) for stmt, l in zip(stmts, labels)]
                a=a.reshape(size)
                mask = mask.reshape(size)
                timers = timers + [Timer(stmt=stmt, num_threads = 10, sub_label=f"{size}, full {mask_true_ratio}", label = "Fill", description = l,
                globals = globals()) for stmt, l in zip(stmts, labels)]
                for t in timers:
                    results.append(t.blocked_autorange())

c=Compare(results)
c.print()