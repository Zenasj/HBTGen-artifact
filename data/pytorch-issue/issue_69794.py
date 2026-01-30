import torch.nn as nn

import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR

param = torch.nn.Parameter(torch.ones(2,2))
optim = torch.optim.Adam(params=[param], lr=0.0004)
warmup_scheduler = LinearLR(optim, start_factor=0.05, end_factor=1, total_iters=10)
decay_scheduler = ExponentialLR(optim, gamma=0.99)
lr_scheduler = SequentialLR(optim, schedulers=[warmup_scheduler, decay_scheduler], milestones=[10])

for _ in range(100):
    optim.zero_grad()
    optim.step()
    lr_scheduler.step()
    print(lr_scheduler.get_last_lr())

self._last_lr = [group['lr'] for group in self.optimizer.param_groups]