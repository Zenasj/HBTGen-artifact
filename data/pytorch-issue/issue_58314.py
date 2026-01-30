import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch


for i in range(1000):
    x = torch.randn(2047, device=torch.device("cpu"), dtype=torch.double)
    _, inds = torch.topk(x, 2000, largest=False)  # Ok
print("No issues when using the CPU")


for i in range(1000):
    print(i)
    x = torch.randn(2047, device=torch.device("cuda"), dtype=torch.double)
    _, inds = torch.topk(x, 2000, largest=False)  # Not Ok
    torch.cuda.synchronize()