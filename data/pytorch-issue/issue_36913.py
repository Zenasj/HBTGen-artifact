import torch
import torchvision

from torch import optim
from torchvision.models import resnet50
from torch.optim import lr_scheduler


model = resnet50()
optimizer = optim.SGD(model.parameters(), lr=0.05)
ms_scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2,4])
lrs = [param_group['lr'] for param_group in optimizer.param_groups]
print(f"lr after lrs= {lrs}")
ms_scheduler.step(epoch=0)
lrs = [param_group['lr'] for param_group in optimizer.param_groups]
print(f"lr after lrs= {lrs}")

model = resnet50()
optimizer = optim.SGD(model.parameters(), lr=0.05)
ms_scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2,4])
lrs = [param_group['lr'] for param_group in optimizer.param_groups]
print(f"lr at 0: {lrs}")

for i in range(10):
  lrs = [param_group['lr'] for param_group in optimizer.param_groups]
  print(f"lr at {i}: {lrs}")
   # train here
  ms_scheduler.step(epoch=None)

model = resnet50()
optimizer = optim.SGD(model.parameters(), lr=0.05)
ms_scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2,4])
lrs = [param_group['lr'] for param_group in optimizer.param_groups]
print(f"lr at 0: {lrs}")

for i in range(1,10):
  lrs = [param_group['lr'] for param_group in optimizer.param_groups]
  print(f"lr at {i}: {lrs}")
   # train here
  ms_scheduler.step(epoch=i)