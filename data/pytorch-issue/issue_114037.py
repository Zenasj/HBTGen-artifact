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
        self.batchnorm2d = nn.BatchNorm2d(3)
    def forward(self, x):

        x = self.batchnorm2d(x)
        x_ = torch.linalg.slogdet(x)
        x_erfinv = torch.special.erfinv(x)
        return x_erfinv

real_inputs = torch.rand(2, 3, 8, 8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(torch.isnan(output_gpu).any())
print(torch.isnan(output_gpu).any())