import random

import torch
import numpy as np


def _inf_nan_preprocess(t, t_np):
    t_np = np.nan_to_num(t_np)
    return t, t_np


@torch.compile
def fn():
    # shape, dims format
    test_cases = (
        (3, 3),
        (4, 4),
        (5, 5),
    )

    for shape in test_cases:
        t = torch.randn(shape, dtype=torch.complex64)
        t_np = np.random.randn(*shape).astype(np.complex64)

        _, t_np = _inf_nan_preprocess(t, t_np)
        print(t, t_np)  # Just a side effect so that compilation kicks in


fn()