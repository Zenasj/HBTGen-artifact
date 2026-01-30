import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@torch.compile
def func(x):
    return x * x

x = torch.tensor(0.0, device="cuda", requires_grad=True)

func(x).backward()
print(x.grad)