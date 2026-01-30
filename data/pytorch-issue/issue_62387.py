import torch

model = models.resnet18()
device = torch.device("cpu")
model.to(device)