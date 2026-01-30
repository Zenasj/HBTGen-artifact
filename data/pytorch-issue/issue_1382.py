import torch
import sys
print(sys.version_info)
print(torch.__version__)
normalcpu = torch.zeros(100, dtype=torch.float32, device="cpu", requires_grad=False).normal_(0., 1.)
h_cpu = torch.histc(normalcpu, bins=10)
normalcuda = normalcpu.to(device="cuda")
h_cuda = torch.histc(normalcuda, bins=10)
print(h_cpu, h_cpu.dtype)
print(h_cuda, h_cuda.dtype)