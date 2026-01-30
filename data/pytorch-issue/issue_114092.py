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
        # = x.view(x.size(0), -1)
        output = F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))
        return output

transform = transforms.Compose([transforms.ToTensor()])
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=64, shuffle=True)

    # 从数据集中获取一个批次的图像作为输入
real_inputs, real_targets = next(iter(data_loader))
model = PreprocessAndCalculateModel()
x = real_inputs
output_gpu = model.to(gpu_device)(x.cuda())
output_cpu = model.to(cpu_device)(x.cpu())
print(torch.allclose(output_gpu.cpu(), output_cpu, atol=1e-1))