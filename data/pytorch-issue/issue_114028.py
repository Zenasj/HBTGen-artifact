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
        # = x.view(x.size(0), -1
        x = x.neg_()
        output = F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))
        return output


real_inputs = torch.rand(2, 3, 8, 8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(torch.allclose(output_gpu.cpu(), output_cpu, atol=1e-1))