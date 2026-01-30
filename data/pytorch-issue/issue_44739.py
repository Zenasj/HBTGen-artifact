import torch
from torch.utils._benchmark import Timer
from torch.utils._benchmark import Compare
import sys

results = []
num_threads = 1
sizes = (100, 1000, 10000)
for size in sizes:
    print(size)
    m = torch.randn(size, size, device="cpu")
    v = torch.randn(size)

    tasks = [("torch.mv(m.t(), v)", "mv"),
               ("(torch.mm(m.t(), v.unsqueeze(-1))).squeeze(-1)", "mm")]
    timers = [Timer(stmt=stmt, num_threads=num_threads, label="mv_backward", sub_label=f"{size}",
    description=label, globals=globals()) for stmt, label in tasks]
    repeats = 3

    for i, timer in enumerate(timers * repeats):
        results.append(
            timer.blocked_autorange()
        )
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
print()

comparison = Compare(results)
comparison.print()