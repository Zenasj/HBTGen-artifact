import torch.nn as nn

import torch


device = torch.device("cuda")
start_cuda_memory = torch.cuda.memory_allocated(device)

def f():
    model = torch.nn.Linear(1, 1, device=device)
    b = torch.randn(1, 1, device=device)
    model(b)

f()

torch.cuda.synchronize(device)
cuda_memory = torch.cuda.memory_allocated(device)
assert cuda_memory <= start_cuda_memory, (cuda_memory, start_cuda_memory)
print("Good!")