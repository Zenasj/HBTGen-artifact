import torch
import psutil
import random
from time import time

bs = 32
data = [torch.randn(3, 128, 128) for _ in range(1000)] # Replay buffer

N = 10
total_time = 0
total_usage = 0.
for _ in range(N):
    s = random.sample(data, bs)
    t0 = time()
    x = torch.stack(s)
    total_time += time() - t0
    total_usage += psutil.cpu_percent(0.1)
    ... # Other things: Forward + Backward + Step

print('Time per call:', 1000 * total_time / N, 'ms')
print('Average usage:', total_usage / N, '%')

import torch
import psutil
import random
from time import time

torch.set_num_threads(1)

bs = 32
data = [torch.randn(3, 128, 128) for _ in range(1000)] # Replay buffer

N = 10
total_time = 0
total_usage = 0.
for _ in range(N):
    s = random.sample(data, bs)
    t0 = time()
    x = torch.stack(s)
    total_time += time() - t0
    total_usage += psutil.cpu_percent(0.1)
    ... # Other things: Forward + Backward + Step

print('Time per call:', 1000 * total_time / N, 'ms')
print('Average usage:', total_usage / N, '%')

import torch
import psutil
import random
from time import time
from torch.utils.data import DataLoader

bs = 32
data = [torch.randn(3, 128, 128) for _ in range(1000)] # Replay buffer

dl = iter(DataLoader(data, bs, True))

N = 10
total_time = 0
total_usage = 0.
for _ in range(N):
    s = random.sample(data, bs)
    t0 = time()
    x = next(dl)
    total_time += time() - t0
    total_usage += psutil.cpu_percent(0.1)
    ... # Other things: Forward + Backward + Step

print('Time per call:', 1000 * total_time / N, 'ms')
print('Average usage:', total_usage / N, '%')

if __name__ == "__main__":
    import torch
    import psutil
    import random
    from time import time
    from torch.utils.data import DataLoader

    bs = 32
    data = [torch.randn(3, 128, 128) for _ in range(1000)] # Replay buffer

    dl = iter(DataLoader(data, bs, True, num_workers=1))

    N = 10
    total_time = 0
    total_usage = 0.
    for _ in range(N):
        s = random.sample(data, bs)
        t0 = time()
        x = next(dl)
        total_time += time() - t0
        total_usage += psutil.cpu_percent(0.1)
        ... # Other things: Forward + Backward + Step

    print('Time per call:', 1000 * total_time / N, 'ms')
    print('Average usage:', total_usage / N, '%')

import multiprocessing as mp
mp.set_start_method('spawn')

import torch
import random
from time import time

bs = 32
data = [torch.randn(3, 128, 128) for _ in range(1000)]

N = 10000
total_time = 0
for _ in range(N):
    s = random.sample(data, bs)
    t0 = time()
    x = torch.stack(s)
    total_time += time() - t0

print('Total Time:', total_time, 'sec')