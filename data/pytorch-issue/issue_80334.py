import torch.nn as nn

from itertools import product

import torch
from torch.utils.benchmark import Compare, Timer


def get_timer(size, device, fn, name):
    # a is in the log-space, i.e. [-inf, 0]
    # b is a probability measure, i.e. [0, 1]
    a = torch.randn(*size, device=device)
    a = -torch.nn.functional.softplus(a)
    a.requires_grad_()

    b = torch.randn(*size, device=device).sigmoid()
    b.requires_grad_()

    z = torch.nn.functional.kl_div(a, b, reduction="batchmean")
    print(name, a.shape, device)

    timer = Timer(
        fn,
        globals={"a": a, "b": b, "z": z},
        label=f"kl_div {device}",
        sub_label=name,
        description="master",
        num_threads=4)

    return timer.blocked_autorange(min_run_time=5)


def get_params():
    fns = [("fwd", "torch.nn.functional.kl_div(a, b, reduction='batchmean')"),
           ("bwd", "z.backward(retain_graph=True)"),
           ("fwd+bwd", "torch.nn.functional.kl_div(a, b, reduction='batchmean').backward()")]
    for device, (name, fn) in product(("cpu", "cuda"), fns):
        yield (1000, 1000), device, fn, name


compare = Compare([get_timer(*params) for params in get_params()])
compare.trim_significant_figures()
compare.print()