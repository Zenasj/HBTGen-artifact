import torch
import pickle
import itertools
from functools import partial
from torch.utils.benchmark import Timer, Compare

benchmark_name = "linalg.lu_factor CUDA"
name = "magma_looped"
label = "lu_factor_{}".format(name)
shapes = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
batches = [(1,), (2,), (4,), (8,), (16,), (32,), (64,), (128,), (512,), (1024,)]
results = []
make_arg = partial(torch.randn, dtype=torch.float32, device="cuda")


for n, batch in itertools.product(shapes, batches):
    if n == 1024 and batch[0] >= 128:
        continue
    if n == 2048 and batch[0] >= 64:
        continue
    A = make_arg(batch + (n, n))
    print(A.shape)
    stmt = "torch.linalg.lu_factor_ex(A)"
    timer = Timer(stmt,
                  globals=globals(),
                  label=benchmark_name,
                  description=label,
                  sub_label=f"shape {A.shape}",
                  num_threads=1)
    results.append(timer.blocked_autorange())

compare = Compare(results)
compare.trim_significant_figures()
compare.print()

with open(f"{label}.pickle", 'wb') as f:
    pickle.dump(results, f)