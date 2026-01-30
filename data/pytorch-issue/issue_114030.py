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
        self.channel_shuffle = nn.ChannelShuffle(groups=3)
    def forward(self, x):
        # = x.view(x.size(0), -1
        output = self.channel_shuffle(x)
        return output


real_inputs = torch.rand(2, 3, 8, 8)
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())