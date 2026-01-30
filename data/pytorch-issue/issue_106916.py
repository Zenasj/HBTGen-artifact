import torch
from torch.testing._internal.common_utils import TestCase

param = torch.rand(2, 3, dtype=torch.float, device='cuda:0', requires_grad=True)
param.grad = torch.rand_like(param)

lr = torch.tensor(.001, device='cuda:0')
opt = torch.optim.Adam([param], lr=lr, fused=True)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    for _ in range(1000):
        opt.step()

print(p.key_averages().table(sort_by="cpu_time_total"))