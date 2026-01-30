import torch.nn as nn
import torchvision

import torch.optim as optim
from torch import nn

conv = nn.Conv2d(3,3,3)
optimizer = optim.Adam(conv.parameters()) 
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2)

# Scheduler with sometimes-constant epoch number
for epoch in [0, 0, 1, 1, 2, 2, 3, 3]:
  lr_scheduler.step(epoch)
  print(optimizer.param_groups[0]['lr'])

import torch.optim as optim
from torch import nn

conv = nn.Conv2d(3,3,3)
optimizer = optim.Adam(conv.parameters()) 
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2)

last_epoch = -1
for epoch in [0, 0, 1, 1, 2, 2, 3, 3]:

  # Check if epoch number has changed manually
  if epoch-last_epoch > 0:
    lr_scheduler.step()
  last_epoch = epoch

  print(epoch, scheduler.get_computed_values())

import torch
from torchvision.models import resnet18
net = resnet18()

optimizer = torch.optim.SGD(net.parameters(), 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

for i in range(10):
  # Scheduler computes and returns new learning rate, leading to unexpected behavior
  print(i, scheduler.get_lr())
  scheduler.step()

import torch
from torchvision.models import resnet18

net = resnet18()
optimizer = torch.optim.SGD(net.parameters(), 0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

for i in range(10):
    # Returns last computed learning rate by scheduler
    print(i, lr_scheduler.get_computed_values())
    lr_scheduler.step()

import torch.optim as optim
from torch import nn

conv = nn.Conv2d(3,3,3)
optimizer = optim.Adam(conv.parameters()) 
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2)

for epoch in range(10):
  # ... some large amount of code ...
  lr_scheduler.step(epoch)
  # ... some large amount of code ...
  lr_scheduler.step(epoch)
  # ... some large amount of code ...