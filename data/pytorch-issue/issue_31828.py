import torch.nn as nn

from torch import nn
import torch
net = nn.Linear(30, 10)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
s = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.1)
print(s.get_lr())
s.step(1)
print(s.get_lr())