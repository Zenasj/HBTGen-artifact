import numpy as np

scipy.linalg.eigh(XTX, b = np.eye(len(XTX), dtype = X.dtype), turbo = True, check_finite = False)

scipy.linalg.eigh(XTX, b = None, turbo = True, check_finite = False)

import torch
from torch.utils.benchmark import Compare, Timer


def get_timer(device, shape):
    A = torch.randn(shape, device=device)

    timer = Timer(
        r"""
        torch.linalg.svd(A, full_matrices=False)""",
        globals={"A": A},
        label="SVD",
        description=f"{device}",
        sub_label=f"shape {shape}",
        num_threads=1,
    )
    return timer.blocked_autorange(min_run_time=5)


def get_params():
    shapes = ((16000, 2, 2),
              (1140, 1140),
              (1534, 1534))
    for device in ("cpu", "cuda"):
        for shape in shapes:
            yield device, shape


compare = Compare([get_timer(*params) for params in get_params()])
compare.trim_significant_figures()
compare.print()