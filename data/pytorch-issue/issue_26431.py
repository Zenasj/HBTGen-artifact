import timeit

for n, t in [(10_000, 8000),
             (100_000, 800)]:
    for dtype in ('torch.float', 'torch.double'):
        print(f'================ dtype {dtype}, {t} times ================================')
        for op in ('sin', 'sinh', 'cos', 'cosh', 'tan'):
            print(f'a.{op}() (a.numel() == {n}) for {t} times')
            print(timeit.timeit(f'a.{op}()',
                                setup=f'import torch; a = torch.arange({n}, device="cpu", dtype={dtype})',
                                number=t))