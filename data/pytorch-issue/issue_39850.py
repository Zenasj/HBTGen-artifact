import torch

sizes = [
    # These will succeed.
    10000,
    100000,
    6042451,
    6042453,
    7000000,

    # This will fail.
    6042452,
]


for n in sizes:
    print(f"n = {n}")
    x = torch.rand((n,), dtype=torch.float64, device="cuda")
    for i in range(100):
        torch.topk(x, k=1, dim=0)
        print(f"\r  {i}", end="")
    print()

import torch
from utils import Timer

torch.manual_seed(0)

experiments = (
    (1, 10000, lambda: torch.rand(size=(39, 222075), device="cuda")),
    (1, 10000, lambda: torch.rand(size=(32, 262144), device="cuda")),
    (1, 4,     lambda: torch.rand(size=(39, 222075), device="cuda")),
    (1, 4,     lambda: torch.rand(size=(32, 262144), device="cuda")),

    (0, 10000, lambda: torch.rand(size=(786842, 25), device="cuda")),
    (0, 10000, lambda: torch.rand(size=(1048576, 16), device="cuda")),
    (0, 4,     lambda: torch.rand(size=(786842, 25), device="cuda")),
    (0, 4,     lambda: torch.rand(size=(1048576, 16), device="cuda")),
)

for dim, k, tensor_constructor in experiments:
    x = tensor_constructor()
    timer = Timer(
        stmt="torch.topk(x, dim=dim, k=k)",
        globals={"x": x, "dim": dim, "k": k},
        label=f"k:{k:>6}, dim:{dim}, size:{list(x.shape)}",
    )
    measurement = timer.blocked_autorange(min_run_time=5)
    print(f"{measurement.median * 1e6:>10.0f} us{'':>10}{measurement.label}")