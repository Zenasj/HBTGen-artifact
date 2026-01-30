import torch.nn as nn

import torch
params = [torch.nn.Parameter(torch.randn(1, 1))]
optimizer = torch.optim.SGD(params, lr=0.2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,8], gamma=0.1)
for epoch in range(10):
    print(scheduler.get_lr())
    scheduler.step()