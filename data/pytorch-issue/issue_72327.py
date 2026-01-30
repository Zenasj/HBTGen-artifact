py
from itertools import product

import torch
import torch.nn as nn
from torch.utils.benchmark import Compare, Timer


def f(size):
    loss = nn.MarginRankingLoss()
    input1 = torch.randn(size, requires_grad=True)
    input2 = torch.randn(size, requires_grad=True)
    target = torch.randn(size).sign()
    output = loss(input1, input2, target)
    output.backward()
    return output


def get_timer(size, num_threads, device):
    timer = Timer(
        "f(size=size)",
        globals={"f": f, "size": size},
        label=f"MarginRankingLoss {device}",
        description=f"time (us)",
        sub_label=f"size: {size}",
        num_threads=num_threads,
    )

    return timer.blocked_autorange(min_run_time=5)


def get_params():
    sizes = ((100,), (1_000,), (10_000,), (100_000,), (1_000_000,))
    devices = ("cpu", "cuda")

    for size, device in product(sizes, devices):
        threads = (1, 2, 4) if device == "cpu" else (1,)
        for num_threads in threads:
            yield size, num_threads, device


def compare():
    compare = Compare([get_timer(*params) for params in get_params()])
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()


compare()