import torch
import time
from contextlib import contextmanager

@contextmanager
def timeit(name, R=3):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    print(f'{name:20s} {1e6*(time.time() - start)/R:8.0f}us per')

xs = torch.randn(int(100e6),).cuda()

with timeit('max'):
    xs.max()
    
with timeit('max-over-dim'):
    xs.max(0)

with timeit('sort-over-dim'):
    xs.sort(0)[-1]