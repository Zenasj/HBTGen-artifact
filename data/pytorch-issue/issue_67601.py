import torch.nn as nn

py
import torch
model = torch.nn.Linear(2,2)
optim = torch.optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([torch.optim.lr_scheduler.ConstantLR(optim), torch.optim.lr_scheduler.StepLR(optim, step_size=10)])

lr_scheduler.optimizer