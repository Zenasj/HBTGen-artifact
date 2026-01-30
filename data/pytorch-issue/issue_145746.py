import torch.nn as nn

import time
import torch
from torch.profiler import profile, ProfilerActivity


def benchmark(function, dtype=torch.float32, check_numerics=True, print_profile=False):
    device = torch.device("cuda")

    shapes = []
    for p in range(24, 30):
        shape = 1<<p
        shapes.append(shape)

    for shape in shapes:
        for _ in range(6):
            x = torch.randn(shape, device=device, dtype=dtype)
            y = function(x)

        if print_profile:
            x = torch.randn(shape, device=device, dtype=dtype)
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                y = function(x)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        x = torch.randn(shape, device=device, dtype=dtype)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(6):
            y = function(x)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        perf_time = (t2 - t1) / 6

        print(f"{function.__name__}, {dtype}, {shape}, {perf_time}")
        if check_numerics:
            x_cpu = x.cpu()
            y_cpu = function(x_cpu).cuda()
            try:
                torch.testing.assert_allclose(y_cpu, y)
            except AssertionError as error:
                print("An exception occurred:", error)


def main():
    ops = [
            torch.relu,
            torch.sigmoid,
            torch.tanh,
            torch.nn.functional.gelu,
            torch.sin,
            torch.exp,
    ]

    dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
    ]

    for op in ops:
        for dtype in dtypes:
            benchmark(op, dtype=dtype)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()