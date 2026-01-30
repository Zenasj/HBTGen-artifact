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
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
    def forward(self, x):
        # = x.view(x.size(0), -1
        mask = torch.rand_like(x) > 0.5
        x_masked = torch.masked_select(x, mask)
        return x_masked

transform = transforms.Compose([transforms.ToTensor()])
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=8, shuffle=True)

real_inputs, real_targets = next(iter(data_loader))
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(torch.allclose(output_gpu.cpu(), output_cpu, atol=1e-1))

print(torch.allclose(output_gpu.cpu(), output_cpu, atol=1e-1))