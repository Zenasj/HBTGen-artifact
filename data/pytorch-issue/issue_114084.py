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
    def forward(self, x,y):
        out = torch.lu_unpack(x, y, unpack_data=True, unpack_pivots=True)
        return out

real_inputs =  torch.tensor([[11, 22], [33, 44]], dtype=torch.float32)
real_inputs2 = torch.tensor([0, 22], dtype=torch.int32)
#real_inputs = torch.rand(2, 8,8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(real_inputs.cuda(),real_inputs2.cuda())
output_cpu = model.to(cpu_device)(real_inputs.cpu(),real_inputs2.cpu())
print(output_gpu)
print(output_cpu)