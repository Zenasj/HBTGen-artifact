import torch

schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=0)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
      for i in range(4):
          dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
          for _ in range(10):
              fn()
          prof.step()

schedule = torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=0)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
      for i in range(4):
          dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
          for _ in range(10):
              fn()
          prof.step()