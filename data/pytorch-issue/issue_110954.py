import torch
steps = [torch.zeros((), device="cpu") for i in range(1000)]

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    torch._foreach_add_(steps, 1024.1024)

print(p.key_averages().table(sort_by="cpu_time_total"))