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
        output1, output2 = torch.lobpcg(x)
        return [output1, output2]

real_inputs = torch.tensor([[0.0100, 0.0000, 0.0000, 0.0000, 0.1000],
        [0.0000, 0.0100, 0.0000, 0.1000, 0.0000],
        [0.0000, 0.0000, 0.0100, 0.0000, 0.0000],
        [0.0000, 0.1000, 0.0000, 0.0100, 0.0000],
        [0.1000, 0.0000, 0.0000, 0.0000, 0.0100]])
#real_inputs = torch.rand(2, 8,8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(output_gpu)
print(output_cpu)