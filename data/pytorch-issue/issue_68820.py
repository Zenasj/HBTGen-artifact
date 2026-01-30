import torch.nn as nn

import torch

model = [torch.nn.parameter.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = torch.optim.SGD(model, 0.1)
scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
optimizer.step()
scheduler.step()
scheduler.get_last_lr()