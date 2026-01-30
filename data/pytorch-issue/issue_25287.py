import timeit

for n, t in [(10_000, 800000),
             (100_000, 80000)]:
    print(f'a.rsqrt() (a.numel() == {n}) for {t} times')
    print(timeit.timeit(f'a.rsqrt()',
                        setup=f'import torch; a = torch.arange({n}, device="cpu", dtype=torch.float)',
                        number=t))