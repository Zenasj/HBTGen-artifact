import torch

from torch.profiler import profile, record_function, ProfilerActivity

with profile(
activities=[torch.profiler.ProfilerActivity.CUDA],
schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/model'),
profile_memory=False,
record_shapes=True,
with_stack=True,
) as prof:
    for _ in range(10):
        y = torch.randn(1).cuda() + torch.randn(1).cuda()
    prof.step()

print(prof.key_averages())