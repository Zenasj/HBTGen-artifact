import torch
import torch.utils.benchmark as benchmark

device = 'cpu'
for p in torch.linspace(1, 16, 16):
    size = int(2**p)
    a = torch.full((size,), 6, dtype=torch.int32, device=device)
    b = torch.full((size,), 3, dtype=torch.int32, device=device)
    t = benchmark.Timer(
        stmt='a & b',
        globals={'a': a, 'b': b}
    ).blocked_autorange(min_run_time=1)
    print('size {}, median time {}us'.format(size, t.median * 1e6))