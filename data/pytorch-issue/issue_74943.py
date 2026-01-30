import torch
import time

inps = [torch.randn(64, 64, 50, 50, device='cuda') for _ in range(2)]

torch.cuda.synchronize()
begin = time.time()
for _ in range(5):
  inps[0].add_(inps[1])
torch.cuda.synchronize()
print(time.time()-begin)

torch.cuda.synchronize()
begin = time.time()
for _ in range(5):
  torch.ops.aten.add_(*inps)
torch.cuda.synchronize()
print(time.time()-begin)

0.005440473556518555
0.7287805080413818

from torch.profiler import profile, record_function, ProfilerActivity

inps = [torch.randn(64, 64, 50, 50, device='cpu') for _ in range(2)]

with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
  torch.ops.aten.add_(*inps)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

for overload in all_overloads:
  try:
    inputs = overload.parse_args(inp)
  except:
    pass
  else:
    break
  call_function(inputs)