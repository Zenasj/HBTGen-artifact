import torch
import time
num_elements = 100000
timed_iters = 100
a = torch.rand(num_elements, dtype=dtype, device=device)
b = torch.rand(1, dtype=dtype, device=device)[0]
c = torch.rand(num_elements, dtype=dtype, device=device)
torch.cuda.synchronize()

start_time = time.time()
for i in range(timed_iters):
    res = a.fmod(b)
torch.cuda.synchronize()
total_time = time.time() - start_time
time_per_iter = total_time / timed_iters