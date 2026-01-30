import torch.nn as nn

self.optimizer = optimizer

import torch
model = torch.nn.Linear(2,2)
optim = torch.optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [torch.optim.lr_scheduler.ConstantLR(optim), torch.optim.lr_scheduler.StepLR(optim, step_size=10)], [1])

lr_scheduler.optimizer