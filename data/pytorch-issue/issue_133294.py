import torch
model = torch._export.aot_load('resnet18_cpu.so', 'cpu')
print(model(torch.ones(4, 3, 224, 224)))