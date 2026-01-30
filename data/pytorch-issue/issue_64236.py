import torch.nn as nn

import torch
from contextlib import contextmanager
import time


@contextmanager
def _log_time_usage(prefix=""):
    '''log the time usage in a code block
    prefix: the prefix text to show
    '''
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.2f" % (end - start))
        print(f'batch={prefix}: elapsed seconds: {elapsed_seconds}')


def compute_and_time_softmax2(tensor, batch_size, repeat):
    with _log_time_usage(batch_size):
        # repeat so that the execution time is more reliable
        for _ in range(repeat):
            torch.nn.Softmax(dim=-1)(tensor)


device = 'cuda'

for batch_size in [1, 4, 16, 64, 256, 1024]:
    compute_and_time_softmax2(torch.randn([batch_size, 1, 50257]).to(device), batch_size, repeat=10000)

# when the input tensors are large, softmax quickly slows down
# batch= 1: elapsed seconds: 0.49
# batch=4: elapsed seconds: 0.36
# batch=16: elapsed seconds: 0.55
# batch=64: elapsed seconds: 2.05
# batch=256: elapsed seconds: 8.25
# batch=1024: elapsed seconds: 31.92

import torch.utils.benchmark as benchmark

def softmax(x):
    torch.nn.Softmax(dim=-1)(x)

dim = 128
for batch_size in [1, 4, 16, 64, 256, 1024]:
    print(" - - - - - ")
    x = torch.randn([batch_size, 1, dim]).to(device)
    t0 = benchmark.Timer(
        stmt='softmax(x)',
        setup='from __main__ import softmax',
        globals={'x': x})
    print(f'batch-size= {batch_size}, x.size(): {x.size()} -> time for softmax(x):  {t0.timeit(100)}')

# here the run times remain flat: 
# batch-size= 1, x.size(): torch.Size([1, 1, 128]) -> time for softmax(x):  <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21610>
# softmax(x)
#   26.51 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 4, x.size(): torch.Size([4, 1, 128]) -> time for softmax(x):  <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21510>
# softmax(x)
#   26.74 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 16, x.size(): torch.Size([16, 1, 128]) -> time for softmax(x):  <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21610>
# softmax(x)
#   27.07 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 64, x.size(): torch.Size([64, 1, 128]) -> time for softmax(x):  <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21510>
# softmax(x)
#   28.33 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 256, x.size(): torch.Size([256, 1, 128]) -> time for softmax(x):  <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21610>
# softmax(x)
#   28.91 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 1024, x.size(): torch.Size([1024, 1, 128]) -> time for softmax(x):  <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21510>
# softmax(x)
#   28.20 us
#   1 measurement, 100 runs , 1 thread



dim = 50257
for batch_size in [1, 4, 16, 64, 256, 1024]:
    print(" - - - - - ")
    x = torch.randn([batch_size, 1, dim]).to(device)
    t0 = benchmark.Timer(
        stmt='softmax(x)',
        setup='from __main__ import softmax',
        globals={'x': x})
    print(f'batch-size= {batch_size}, x.size(): {x.size()} -> time for softmax(x): {t0.timeit(100)}')

# here run times quickly increase 
# batch-size= 1, x.size(): torch.Size([1, 1, 50257]) -> time for softmax(x): <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21610>
# softmax(x)
#   102.26 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 4, x.size(): torch.Size([4, 1, 50257]) -> time for softmax(x): <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21510>
# softmax(x)
#   103.18 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 16, x.size(): torch.Size([16, 1, 50257]) -> time for softmax(x): <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21610>
# softmax(x)
#   122.07 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 64, x.size(): torch.Size([64, 1, 50257]) -> time for softmax(x): <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21510>
# softmax(x)
#   242.17 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 256, x.size(): torch.Size([256, 1, 50257]) -> time for softmax(x): <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21610>
# softmax(x)
#   966.73 us
#   1 measurement, 100 runs , 1 thread
#  - - - - - 
# batch-size= 1024, x.size(): torch.Size([1024, 1, 50257]) -> time for softmax(x): <torch.utils.benchmark.utils.common.Measurement object at 0x7f111bc21510>
# softmax(x)
#   3.55 ms
#   1 measurement, 100 runs , 1 thread