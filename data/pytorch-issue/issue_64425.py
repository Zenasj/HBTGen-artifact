import torch
import time
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare
import sys

print('Using pytorch %s' % (torch.__version__))

shapes = [59, 60, 128, 256, 512, 1024, 2048, 4096, 8192]
results = []
num_threads = 1
dtype = torch.float32
repeats = 2

for shape in shapes:
    a = torch.randn(shape, shape)
    b = torch.randn(shape, shape)
    b.transpose_(-2, -1)
    assert b.transpose(-2, -1).is_contiguous()

    av = a.clone().reshape(1, shape, shape)
    bv = b.clone().reshape(1, shape, shape)
    assert bv.transpose(-2, -1).is_contiguous()

    tasks = [("av.copy_(bv)", "copy"), ("a.view(1,shape,shape).copy_(b.view(1,shape,shape))", "copy_view"), ("a.copy_(b)", "transpose_copy"),]
    timers = [Timer(stmt=stmt, num_threads=num_threads, label=f"copy", sub_label=f"{a.shape}", description=label, globals=globals()) for stmt, label in tasks]

    for i, timer in enumerate(timers * repeats):
        results.append(
            timer.blocked_autorange()
        )
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()

compare = Compare(results)
# compare.trim_significant_figures()
compare.colorize()
compare.print()