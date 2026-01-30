import timeit

dtypes = [
    'float16',
    'float32',
    'float64',
]

shapes = [
    (1000, 1000),
    (1000, 2000),
    (1000, 4000),
    (1000, 8000),
    (1000, 16000),
]

device = 'cuda'
number = 1000

def run(shape, dtype, printit=True):
    print(timeit.timeit(
        setup=f"import torch; a = torch.rand(*{shape}, dtype=torch.{dtype}, device='{device}');",
        stmt=f"torch.multinomial(a, num_samples=1, replacement=True)",
        number=number
    ))

for shape in shapes:
    for dtype in dtypes:
        run(shape, dtype)

import torch.utils.benchmark as benchmark
import torch

def run(a):
    torch.multinomial(a, num_samples=1, replacement=True)

dtypes = [
    torch.float16,
    torch.float32,
    torch.float64,
]

shapes = [
    (1000, 1000),
    (1000, 2000),
    (1000, 4000),
    (1000, 8000),
    (1000, 16000),
]

device = 'cuda'
number = 1000

for shape in shapes:
    for dtype in dtypes:
        a = torch.rand(*shape, dtype=dtype, device=device)
        m = benchmark.Timer(
            setup=f"from __main__ import run",
            stmt=f"run(a)",
            globals={"a": a},
        ).blocked_autorange(min_run_time=1)
        print(f"| {shape} | {dtype} | {m.mean * 1000:.3f} |")

torch.multinomial(torch.rand(2, 3, dtype=torch.float16, device="cuda"), num_samples=1, replacement=True)