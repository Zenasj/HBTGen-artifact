import timeit

for n, t in [(1000, 13000),
            (10_000, 1300)]:
    for e in (2, 3, 4):
        for dtype in ('torch.int16', 'torch.int32', 'torch.int64'):
            print(f'a.pow({e}) (a.numel() == {n}) for {t} times')
            print(f'dtype {dtype}, {t} times', end='\t\t')
            print(timeit.timeit(f'a.pow({e})',
                                setup=f'import torch; a = torch.arange({n}, device="cpu", dtype={dtype})',
                                number=t))