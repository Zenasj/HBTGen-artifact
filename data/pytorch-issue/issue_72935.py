import torch
import pickle
import itertools
from functools import partial
from torch.utils.benchmark import Timer, Compare

name = "heuristic"
label = "lu_solve {}".format(name)
shapes = [1, 2, 8, 16, 32, 64, 128, 256]
batches = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,), (512,), (1024,)]
results = []
make_arg = partial(torch.randn, dtype=torch.float32, device="cuda")

for n, batch in itertools.product(shapes, batches):
    LU, pivots = torch.linalg.lu_factor(make_arg(batch + (n, n)))
    B = make_arg(batch + (n, 1))
    print(LU.shape)
    stmt = "torch.linalg.lu_solve(LU, pivots, B, adjoint=adjoint)"
    for adjoint in (True, False):
        timer = Timer(stmt,
                      globals=globals(),
                      label="linalg.lu_solve CUDA{}".format(" Adjoint" if adjoint else ""),
                      description=label,
                      sub_label=f"shape {LU.shape}",
                      num_threads=1)
        results.append(timer.blocked_autorange())


compare = Compare(results)
compare.trim_significant_figures()
compare.print()

with open("{}_lu_solve.pickle".format(name), 'wb') as f:
    pickle.dump(results, f)

import pickle
from torch.utils.benchmark import Timer, Compare

files = [
         "looped_magma",
         "looped cusolver",
         "batched cublas",
         "batched magma",
         "unpack+solve_triangular",
         "heuristic",
        ]

timers = []
for name in files:
    with open("{}_lu_solve.pickle".format(name), 'rb') as f:
        timers += pickle.load(f)

compare = Compare(timers)
compare.trim_significant_figures()
compare.print()