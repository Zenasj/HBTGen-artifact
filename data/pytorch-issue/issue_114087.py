import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
import torch.nn.functional as F
gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")
class PreprocessAndCalculateModel(nn.Module):
    def init(self):
        super().init()
    def forward(self, x):
        out = torch.mm(x, x, out=x)
        return out

real_inputs = torch.randn(32,32)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(real_inputs.cuda())
output_cpu = model.to(cpu_device)(real_inputs.cpu())
print(output_gpu)
print(output_cpu)