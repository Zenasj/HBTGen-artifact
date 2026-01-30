import torch.nn as nn
import torchvision

import torch
from torch.nn import Parameter

optimizer = torch.optim.SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

for i in range(10):
  print(i, [group['lr'] for group in optimizer.param_groups])  
  scheduler.step()

for i in range(10):
  print(i, scheduler.get_lr())  # Deprecation warning
  scheduler.step()

for i in range(10):
  print(i, scheduler.step())

import torch.optim as optim
from torch import nn

conv = nn.Conv2d(3,3,3)
optimizer = optim.Adam(conv.parameters()) 
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2)

for _ in range(10):
    # NOTE Remove epoch parameter
    lr_scheduler.step()
    print(optimizer.param_groups[0]['lr'])

def compute_using_closed_form(epoch):
    return 0.5**epoch

for epoch in range(10):
    # NOTE Invoke user-defined closed form
    lr = compute_using_closed_form(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(optimizer.param_groups[0]['lr'])

lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**epoch)

for _ in range(10):
    lr_scheduler.step()
    print(lr_scheduler.get_computed_values())

import torch
from torchvision.models import resnet18

net = resnet18()
optimizer = torch.optim.SGD(net.parameters(), 0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

for i in range(10):
    # NOTE Invoke new interface to get values
    print(i, lr_scheduler.get_computed_values())
    lr_scheduler.step()