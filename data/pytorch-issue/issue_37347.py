import torch
import torch.nn as nn

linear_parameters = list(torch.nn.Linear(1, 1).parameters())
optimizer = torch.optim.Adam([
    {'params': linear_parameters},
    {'params': list(reversed(linear_parameters))}
])

linear_parameters = list(torch.nn.Linear(1, 1).parameters())
optimizer = torch.optim.Adam([{'params': linear_parameters}])
optimizer.param_groups.extend(optimizer.param_groups)