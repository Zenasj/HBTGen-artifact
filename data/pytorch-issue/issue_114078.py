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
        output = torch.arccosh(x)
        return output

#real_inputs = torch.tensor()
real_inputs = torch.rand(2, 3, 8, 8,8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
#print(output_gpu)
#print(output_cpu)
#print(torch.isnan(output_gpu).any())
#print(torch.isnan(output_cpu).any())
print(torch.allclose(output_gpu.cpu(), output_cpu, atol=1))