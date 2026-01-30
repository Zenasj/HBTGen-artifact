import torch
torch.zeros(2).cuda(0)

import torch
print(torch.__version__)
torch.zeros(2).cuda(0)

import torch

print(f"Pytorch version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
device = torch.device('cuda')
print(f"A torch tensor: {torch.rand(5).to(device)}")

import torch
torch.cuda.set_device(0)
torch.cuda.set_device(1)
x = torch.zeros(10000, 100000).cuda(1)
x = torch.zeros(10000, 100000).cuda(0)