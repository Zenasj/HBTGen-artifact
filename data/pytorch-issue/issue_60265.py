import torch.nn as nn

import torch

parameter = torch.nn.Parameter(torch.zeros(10), requires_grad=True)

base_lr = 0.001

epoch_end = 100

optimizer = torch.optim.Adam([parameter], lr=base_lr)
print('1', optimizer.param_groups[0]['lr'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epoch_end
)
print('2', scheduler.state_dict())

optimizer.step()
print('3', optimizer.param_groups[0]['lr'])
scheduler.step()
print('4', optimizer.param_groups[0]['lr'])
print('5', scheduler.state_dict())

new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epoch_end, last_epoch=0
)
print('6', new_scheduler.state_dict())

import torch

parameter = torch.nn.Parameter(torch.zeros(10), requires_grad=True)

base_lr = 0.001
epoch_start = 0
epoch_end = 100

optimizer = torch.optim.Adam([parameter], lr=base_lr)
print('1', optimizer.param_groups[0]['lr'])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epoch_end
)
print('2', scheduler.state_dict())

optimizer.step()
print('3', optimizer.param_groups[0]['lr'])
scheduler.step()
print('4', optimizer.param_groups[0]['lr'])
print('5', scheduler.state_dict())

optimizer_state = optimizer.state_dict()
new_optimizer = torch.optim.Adam([parameter], lr=base_lr)
new_optimizer.load_state_dict(optimizer_state)
print('6', new_optimizer.param_groups[0]['lr'])

new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    new_optimizer, T_max=epoch_end, last_epoch=0
)
print('7', new_optimizer.param_groups[0]['lr'])