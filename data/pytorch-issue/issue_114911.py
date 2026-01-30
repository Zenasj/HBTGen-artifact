import torch.nn as nn

import torch
import time
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
x = torch.randn(2, 128, 65536).cuda()
def forward(x):
    return torch.einsum("s b k, t b k -> ", x, x)
def benchmark_fn(name, fn, *args, **kwargs):
    for _ in range(5):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(100):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    dt = (time.time() - begin)
    dt_us = int(dt * 1000000) / 100
    print(f"{name}:", dt_us, "us")
print("torch: ", torch.__version__)
benchmark_fn("fn", forward, x)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     forward(x)
# prof.export_chrome_trace("einsum_nightly_conda_cu121.pt.trace.json.gz")

import torch
import time


def benchmark_fn(name, fn, args, warmup=5, cycles=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(cycles):
        fn(*args)
    torch.cuda.synchronize()
    dt = (time.time() - begin)
    dt_us = int(dt * 1000000) / cycles
    print(f"{name}:", dt_us, "us")


if __name__ == "__main__":
    print("torch: ", torch.__version__, " device: ", torch.cuda.get_device_name(0))
    m, n, k=1, 1, 65535
    a=torch.rand((m, k), device='cuda')
    b=torch.rand((k, n), device='cuda')

    benchmark_fn("bmm", torch.bmm, (a.unsqueeze(0), b.unsqueeze(0)))
    benchmark_fn("mm", torch.mm, (a, b))