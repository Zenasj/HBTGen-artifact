import torch
import os

a = torch.rand((2,2))
b = a.to("cuda")

print("set PYTORCH_NO_CUDA_MEMORY_CACHING=1")
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]="1"
c = torch.rand((2,2))
d  = c.to("cuda")