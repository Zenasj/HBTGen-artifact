import torch.nn as nn

import torch
from torch.utils.benchmark import Timer

x = torch.ones((1, 1))
weights = {
    "Tensor weight": torch.ones((1, 1)),
    "Parameter weight": torch.nn.Linear(1, 1).weight,
}

for label, weight in weights.items():
    timer = Timer(
        "torch.nn.functional.linear(x, weight)",
        globals={"x": x, "weight": weight},
        label=label,
    )
    print(timer.blocked_autorange(min_run_time=2), "\n")