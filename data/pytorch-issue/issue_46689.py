import torch

from torch.utils.benchmark import Timer

timer = Timer(
    "x == y",
    "x = torch.ones((1,)); y = torch.ones((1,))",
)

print(timer.blocked_autorange(min_run_time=1), "\n\n")
print(timer.collect_callgrind())