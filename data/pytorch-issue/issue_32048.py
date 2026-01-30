import torch
from torch.utils.benchmark import Compare, Timer


def get_timer(device, n):
    shape = (n, n)
    A = torch.randn(shape, device=device)

    timer = Timer(
        r"""
        torch.linalg.slogdet(A)""",
        globals={"A": A},
        label="torch.linalg.logdet",
        description=f"{device}",
        sub_label=f"shape {shape}",
        num_threads=1,
    )
    return timer.blocked_autorange()


def get_params():
    ns = (100, 200, 500, 1000, 2000)
    for device in ("cpu", "cuda"):
        for n in ns:
            yield device, n


compare = Compare([get_timer(*params) for params in get_params()])
compare.trim_significant_figures()
compare.print()