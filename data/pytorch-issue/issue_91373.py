import random

import numpy as np
import torch
from torch.utils.benchmark import Timer, Compare


def custom(x, dim):
    return (x*x).sum(dim=1).sqrt()


def custom_np(x, axis):
    return np.sqrt((x*x).sum(axis=1))


fns = (torch.linalg.norm,
       torch.linalg.vector_norm,
       custom)
np_fns = (np.linalg.norm, custom_np)


def gen_inputs():
    shape = (200000, 3)
    for fn in fns:
        t = torch.randn(shape, device="cpu")
        yield (str(tuple(shape)), "torch", fn, t), dict(dim=1)
    for fn in np_fns:
        t = np.random.randn(200000, 3)
        yield (str(tuple(shape)), "numpy", fn, t), dict(axis=1)


def benchmark(description, lib, f, *args, **kwargs):
    return Timer("f(*args, **kwargs)",
                 globals=locals(),
                 label="Norm",
                 sub_label=f"{lib} {f.__name__}",
                 description=description,
                 num_threads=torch.get_num_threads()).blocked_autorange()


results = []
for args, kwargs in gen_inputs():
    results.append(benchmark(*args, **kwargs))

compare = Compare(results)
compare.trim_significant_figures()
compare.print()