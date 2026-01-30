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
        output = torch.pinverse(x)
        return output


real_inputs = torch.tensor([[0.0, 1.0, -1.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(output_gpu)
print(output_cpu)

### Versions