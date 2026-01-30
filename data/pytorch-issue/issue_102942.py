import torch

with torch.profiler.profile(
            with_stack=True,
            profile_memory=True,
            record_shapes=True
        ) as prof:
    x = torch.rand(128, 128, device='cuda')
    x * x + x * x
    plot = profile_plot(prof)