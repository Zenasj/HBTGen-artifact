import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

sz = 65

for i in range(2):
    arr = torch.rand((sz, sz//2 + 1, 2), dtype=torch.float32, device=i)
    # Get the crash
    try:
        torch.irfft(arr, 2, signal_sizes=(sz, sz))
    except RuntimeError as e:
        print(str(i))
        raise e

torch.backends.cuda.cufft_plan_cache.max_size = 0