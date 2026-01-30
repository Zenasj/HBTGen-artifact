import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
import torch.nn.functional as F
gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        device = x.device
        output = torch.geqrf(x)
        return output

real_inputs = torch.Tensor([[1,1,1,1],[1,1,1,1]])
#real_inputs = torch.rand(2, 3, 8, 8,8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(output_gpu)
print(output_cpu)

import torch
import torch.nn as nn
gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        output = torch.geqrf(x)
        return output

real_inputs = torch.Tensor([[1,1,1,1],[1,1,1,1]])
#real_inputs = torch.rand(2, 3, 8, 8,8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print((output_gpu.a.cpu()-output_cpu.a).abs().max())
print((output_gpu.tau.cpu()-output_cpu.tau).abs().max())